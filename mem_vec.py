#!/usr/bin/env python3
"""mem_vec — capa semantica para el CLI `mem` (memorias en SQLite).

Embeddings locales via Ollama (modelo por defecto qwen3-embedding:4b).
Sin dependencias externas: solo stdlib. Los vectores se guardan como BLOB
float32 en la tabla `embeddings` y la busqueda KNN se hace a fuerza bruta
con similitud coseno (instantaneo a esta escala).

Subcomandos:
  reindex [--all]                 (re)genera embeddings de las memorias activas
  search <consulta> [-k N] [--here] [--device D] [--json]
  dupes  --nombre N --resumen R ...   memorias similares a una candidata (dedupe)
  embed-one <texto>               imprime el vector (debug)

Config por entorno:
  MEM_DB            ruta a la DB (default XDG)
  OLLAMA_URL        default http://localhost:11434
  MEM_EMBED_MODEL   default qwen3-embedding:4b
"""
import argparse
import array
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
import urllib.request
from datetime import datetime as _dt

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MEM_EMBED_MODEL", "qwen3-embedding:4b")
# El modelo procesa hasta ~2048 tokens por lote. Truncamos por caracteres
# con margen (espanol ~3.5 char/token). El resumen+nombre van primero, asi
# que aunque se corte el contenido largo, el gist se conserva.
MAX_CHARS = 6000
# Mantener el modelo en VRAM 30 min tras el ultimo uso: durante una sesion de
# trabajo sigue caliente (~0.6s/consulta); se libera solo despues (no compite
# con juegos permanentemente). El arranque en frio es ~47s la primera vez.
KEEP_ALIVE = os.environ.get("MEM_KEEP_ALIVE", "30m")

# Modelo generativo (instruct) para tareas de razonamiento: contradicciones,
# y a futuro resumen/auto-tags. Keep-alive corto: se usa a demanda, no conviene
# tenerlo ocupando VRAM frente a los juegos.
GEN_MODEL = os.environ.get("MEM_GEN_MODEL", "llama3.1:8b")
GEN_KEEP_ALIVE = os.environ.get("MEM_GEN_KEEP_ALIVE", "10m")


class EmbedError(Exception):
    """Fallo al embeber un item concreto (no fatal: se puede saltar)."""


class GenError(Exception):
    """Fallo al consultar el modelo generativo (no fatal: degradar a 'sin juicio')."""


def db_path():
    p = os.environ.get("MEM_DB")
    if p:
        return p
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(xdg, "mem", "memory.db")


def vec_path():
    # Embeddings = datos derivados, locales por equipo. Fuera de git (gitignored).
    # Regenerables con `mem reindex`. NO van en memory.db (que se sincroniza).
    return os.environ.get("MEM_VEC_DB") or os.path.join(
        os.path.dirname(db_path()), "vec.db")


def connect():
    # Abrimos la DB de embeddings y adjuntamos la principal (solo lectura logica).
    con = sqlite3.connect(vec_path())
    # Varios 'reindex'/'preindex' en background pueden escribir a la vez (auto tras
    # remember/add/note/rm): que un escritor que colisiona ESPERE en vez de morir
    # con 'database is locked'. WAL ademas deja leer mientras se escribe.
    con.execute("PRAGMA busy_timeout=5000")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("ATTACH DATABASE ? AS mem", (db_path(),))
    con.execute(
        """CREATE TABLE IF NOT EXISTS embeddings (
             memoria_id  INTEGER PRIMARY KEY,
             modelo      TEXT NOT NULL,
             dim         INTEGER NOT NULL,
             hash        TEXT NOT NULL,
             vec         BLOB NOT NULL,
             actualizado TEXT NOT NULL DEFAULT (datetime('now','localtime'))
           );"""
    )
    # RAG sobre archivos del disco (apuntes/configs/dotfiles). Indice aparte de
    # las memorias: un chunk por fila, brute-force coseno como en search.
    con.execute(
        """CREATE TABLE IF NOT EXISTS archivos (
             id     INTEGER PRIMARY KEY AUTOINCREMENT,
             ruta   TEXT NOT NULL,
             chunk  INTEGER NOT NULL,
             inicio INTEGER,
             texto  TEXT NOT NULL,
             hash   TEXT NOT NULL,
             modelo TEXT NOT NULL,
             dim    INTEGER NOT NULL,
             vec    BLOB NOT NULL,
             mtime  REAL,
             UNIQUE(ruta, chunk)
           );"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_arch_ruta ON archivos(ruta);")
    con.execute("CREATE TABLE IF NOT EXISTS archivos_roots (ruta TEXT PRIMARY KEY);")
    # Embeddings de pedidos (tareas): busqueda semantica de tareas por tema.
    con.execute(
        """CREATE TABLE IF NOT EXISTS pedido_embeddings (
             pedido_id   INTEGER PRIMARY KEY,
             modelo      TEXT NOT NULL,
             dim         INTEGER NOT NULL,
             hash        TEXT NOT NULL,
             vec         BLOB NOT NULL,
             actualizado TEXT NOT NULL DEFAULT (datetime('now','localtime'))
           );"""
    )
    return con


def format_prompt(text, role):
    """Cada familia de modelos espera un formato distinto para query vs documento.

    - qwen3-embedding: query lleva instruccion ('Instruct:...\\nQuery:...'),
      el documento va crudo. Usar prefijos estilo nomic le baja la calidad.
    - nomic-embed-text: prefijos asimetricos 'search_query:' / 'search_document:'.
    - resto (bge-m3, etc.): texto crudo en ambos lados.
    """
    if MODEL.startswith("qwen3-embedding"):
        if role == "query":
            task = "Given a search query, retrieve relevant memories that answer it"
            return f"Instruct: {task}\nQuery: {text}"
        return text
    if MODEL.startswith("nomic-embed"):
        return f"{'search_query' if role == 'query' else 'search_document'}: {text}"
    return text


def embed(text, role):
    """Llama a Ollama con el formato correcto segun el modelo y el rol.

    Lanza EmbedError en fallos por-item (HTTP 500, p.ej. texto muy largo);
    aborta el proceso solo si Ollama esta inaccesible (conexion rechazada).
    """
    budget = min(len(text), MAX_CHARS)
    while True:
        body = json.dumps({"model": MODEL,
                           "prompt": format_prompt(text[:budget], role),
                           "keep_alive": KEEP_ALIVE}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return array.array("f", json.load(r)["embedding"])
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            # texto excede el lote/contexto del modelo -> recortar y reintentar
            too_long = ("too large" in msg or "exceeds the context" in msg
                        or "input length" in msg)
            if e.code == 500 and too_long and budget > 500:
                budget = int(budget * 0.7)
                continue
            raise EmbedError(f"HTTP {e.code} de Ollama: {msg[:200]}")
        except urllib.error.URLError as e:
            sys.exit(f"error: Ollama inaccesible en {OLLAMA_URL}: {e}\n"
                     f"¿esta corriendo el servicio? (systemctl status ollama)")


def doc_text(tipo, nombre, resumen, contenido, tags):
    parts = [tipo or "", nombre or "", resumen or "", contenido or "", tags or ""]
    return "\n".join(p for p in parts if p)


def content_hash(tipo, nombre, resumen, contenido, tags):
    h = hashlib.sha256()
    h.update(doc_text(tipo, nombre, resumen, contenido, tags).encode())
    h.update(MODEL.encode())
    return h.hexdigest()


def cosine(a, b):
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


# ---------- capa generativa (modelo instruct via Ollama) ----------
def gen_json(system, user, timeout=120):
    """Consulta el modelo instruct (Ollama /api/chat) en modo JSON.

    Devuelve un dict. Lanza GenError ante cualquier fallo (Ollama caido,
    modelo ausente -> 404, respuesta no parseable). El llamador degrada con
    gracia: una funcion de auditoria nunca debe romper un flujo principal.
    """
    body = json.dumps({
        "model": GEN_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "format": "json",
        "keep_alive": GEN_KEEP_ALIVE,
        "options": {"temperature": 0, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            content = json.load(r)["message"]["content"]
        out = json.loads(content)
    except (OSError, KeyError, ValueError) as e:
        # OSError cubre URLError, HTTPError (404 si falta el modelo), socket.timeout
        # (== TimeoutError) y conexion rechazada de forma uniforme.
        raise GenError(str(e))
    if not isinstance(out, dict):
        # format=json puede devolver lista/escalar/null si el modelo ignora el schema.
        raise GenError("respuesta JSON no es un objeto")
    return out


_JUDGE_SYS = (
    "Eres un auditor RIGUROSO de una base de conocimiento personal en español. "
    "Clasificas la relación entre dos notas (A y B).\n\n"
    "Definición ESTRICTA: A y B se CONTRADICEN solo si hay un hecho o regla donde "
    "A afirma algo y B afirma lo opuesto o un valor incompatible sobre el MISMO "
    "sujeto, de modo que NO pueden ser ambas verdaderas a la vez.\n\n"
    "NO es contradicción (responde 'consistente'): que traten el mismo tema o se "
    "solapen; que una amplíe, detalle o dé contexto a la otra; que mencionen "
    "aspectos distintos sin chocar; que digan lo mismo con otras palabras.\n"
    "Si no comparten sujeto, responde 'no_relacionado'.\n\n"
    "Primero identifica la afirmación central de cada nota y decide si pueden ser "
    "ambas verdaderas a la vez. Responde SOLO JSON válido:\n"
    '{"afirmacion_a":"<central de A, breve>","afirmacion_b":"<central de B, breve>",'
    '"pueden_ser_ambas_verdaderas":true|false,'
    '"veredicto":"contradice|consistente|no_relacionado","confianza":<0..1>,'
    '"motivo":"<una frase>"}\n\n'
    "Ejemplo contradicción -> A:\"Para deploys siempre usar npm\" "
    "B:\"Nunca usar npm, usar pnpm\" -> pueden_ser_ambas_verdaderas:false, "
    'veredicto:"contradice".\n'
    "Ejemplo consistente (una amplía a la otra) -> A:\"Sudo se aprueba con la app "
    "Panda Control\" B:\"Sudo usa SUDO_ASKPASS apuntando al askpass de la app Panda "
    'Control\" -> pueden_ser_ambas_verdaderas:true, veredicto:"consistente".'
)


def _as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "si", "sí", "yes")


def judge_contradiction(a_name, a_text, b_name, b_text):
    """Devuelve (veredicto, confianza, motivo). Lanza GenError si el modelo falla."""
    user = (f"NOTA A — {a_name}:\n{a_text}\n\n"
            f"NOTA B — {b_name}:\n{b_text}\n\n"
            "Clasifica la relación entre A y B según las reglas.")
    out = gen_json(_JUDGE_SYS, user)
    ver = str(out.get("veredicto", "")).strip().lower()
    try:
        conf = float(out.get("confianza", 0) or 0)
    except (TypeError, ValueError):
        conf = 0.0
    motivo = str(out.get("motivo", "")).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
    # Guardarraíl: si el propio modelo dice que ambas pueden ser verdaderas,
    # NO es contradicción (corrige el sesgo a sobre-marcar de modelos chicos).
    if "pueden_ser_ambas_verdaderas" in out and _as_bool(out["pueden_ser_ambas_verdaderas"]):
        if ver == "contradice":
            ver = "consistente"
    return ver, conf, motivo


def _judge_text(resumen, contenido, limit=800):
    """Texto compacto de una memoria para que lo juzgue el modelo."""
    t = resumen or ""
    if contenido and contenido != resumen:
        t += "\n" + contenido
    return t[:limit]


def related_memories(con, qvec, exclude_name, rel, k):
    """Memorias activas con similitud coseno >= rel al vector dado, top-k,
    excluyendo `exclude_name`. Devuelve (score, tipo, nombre, resumen, contenido, disp)."""
    rows = con.execute(
        "SELECT m.tipo,m.nombre,m.resumen,m.contenido,m.dispositivo,e.vec "
        "FROM mem.memorias m JOIN embeddings e ON e.memoria_id=m.id "
        "WHERE m.activo=1 AND m.nombre<>?", (exclude_name,)).fetchall()
    out = []
    for tipo, nombre, resumen, contenido, disp, vec in rows:
        v = array.array("f")
        v.frombytes(vec)
        if len(v) != len(qvec):
            continue
        s = cosine(qvec, v)
        if s >= rel:
            out.append((s, tipo, nombre, resumen, contenido, disp))
    out.sort(reverse=True, key=lambda x: x[0])
    return out[:k]


def cmd_reindex(args):
    con = connect()
    rows = con.execute(
        "SELECT id,tipo,nombre,resumen,contenido,tags FROM mem.memorias"
        + ("" if args.all else " WHERE activo=1")
    ).fetchall()
    have = {r[0]: r[1] for r in con.execute("SELECT memoria_id,hash FROM embeddings")}
    ids_now = set()
    nuevos = saltados = fallidos = 0
    for mid, tipo, nombre, resumen, contenido, tags in rows:
        ids_now.add(mid)
        h = content_hash(tipo, nombre, resumen, contenido, tags)
        if have.get(mid) == h:
            saltados += 1
            continue
        try:
            vec = embed(doc_text(tipo, nombre, resumen, contenido, tags), "document")
        except EmbedError as e:
            fallidos += 1
            print(f"  SALTADA (no embebida): {nombre} — {e}", file=sys.stderr)
            continue
        con.execute(
            "INSERT INTO embeddings(memoria_id,modelo,dim,hash,vec,actualizado) "
            "VALUES(?,?,?,?,?,datetime('now','localtime')) "
            "ON CONFLICT(memoria_id) DO UPDATE SET "
            "modelo=excluded.modelo,dim=excluded.dim,hash=excluded.hash,"
            "vec=excluded.vec,actualizado=excluded.actualizado",
            (mid, MODEL, len(vec), h, vec.tobytes()),
        )
        con.commit()  # commit incremental: un fallo posterior no pierde lo ya hecho
        nuevos += 1
        print(f"  embebida: {nombre}", file=sys.stderr)
    # limpiar embeddings huerfanos (memorias borradas)
    huerfanos = [k for k in have if k not in ids_now]
    for k in huerfanos:
        con.execute("DELETE FROM embeddings WHERE memoria_id=?", (k,))
    con.commit()
    print(f"reindex: {nuevos} (re)embebidas, {saltados} sin cambios, "
          f"{fallidos} fallidas, {len(huerfanos)} huerfanas eliminadas")


def cmd_search(args):
    con = connect()
    qvec = embed(args.consulta, "query")
    where = "WHERE m.activo=1"
    params = []
    if args.device:
        where += " AND m.dispositivo=?"
        params.append(args.device)
    elif args.here:
        where += " AND (m.dispositivo='compartida' OR m.dispositivo=?)"
        params.append(args.here_device)
    rows = con.execute(
        f"SELECT m.id,m.tipo,m.nombre,m.resumen,m.dispositivo,e.vec,e.dim "
        f"FROM mem.memorias m JOIN embeddings e ON e.memoria_id=m.id {where}",
        params,
    ).fetchall()
    scored = []
    for mid, tipo, nombre, resumen, disp, vec, dim in rows:
        v = array.array("f")
        v.frombytes(vec)
        scored.append((cosine(qvec, v), tipo, nombre, resumen, disp))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[: args.k]
    if args.json:
        print(json.dumps(
            [{"score": round(s, 4), "tipo": t, "nombre": n, "resumen": r, "dispositivo": d}
             for s, t, n, r, d in top], ensure_ascii=False, indent=2))
        return
    if not top:
        print("(sin resultados — ¿corriste 'mem reindex'?)")
        return
    for s, tipo, nombre, resumen, disp in top:
        print(f"[{s:.3f}] [{tipo}] {nombre} — {resumen}")


# ---------- busqueda semantica de PEDIDOS (tareas) ----------
def pedido_doc(titulo, detalle, notas):
    return "\n".join(p for p in [titulo or "", detalle or "", notas or ""] if p)


def pedido_hash(titulo, detalle, notas):
    h = hashlib.sha256()
    h.update(pedido_doc(titulo, detalle, notas).encode())
    h.update(MODEL.encode())
    return h.hexdigest()


def cmd_preindex(args):
    """(Re)genera embeddings de los pedidos. Incremental por hash (incluye MODEL);
    poda huerfanos de pedidos borrados. Auto tras 'mem add/note/rm'."""
    con = connect()
    rows = con.execute("SELECT id,titulo,detalle,notas FROM mem.pedidos").fetchall()
    have = {r[0]: r[1] for r in con.execute("SELECT pedido_id,hash FROM pedido_embeddings")}
    ids_now = set()
    nuevos = saltados = fallidos = 0
    for pid, titulo, detalle, notas in rows:
        ids_now.add(pid)
        doc = pedido_doc(titulo, detalle, notas)
        if not doc.strip():
            continue  # pedido sin texto util: nada que embeber (no es huerfano)
        h = pedido_hash(titulo, detalle, notas)
        if have.get(pid) == h:
            saltados += 1
            continue
        try:
            vec = embed(doc, "document")
        except EmbedError as e:
            fallidos += 1
            print(f"  SALTADO pedido #{pid}: {e}", file=sys.stderr)
            continue
        con.execute(
            "INSERT INTO pedido_embeddings(pedido_id,modelo,dim,hash,vec,actualizado) "
            "VALUES(?,?,?,?,?,datetime('now','localtime')) "
            "ON CONFLICT(pedido_id) DO UPDATE SET "
            "modelo=excluded.modelo,dim=excluded.dim,hash=excluded.hash,"
            "vec=excluded.vec,actualizado=excluded.actualizado",
            (pid, MODEL, len(vec), h, vec.tobytes()))
        con.commit()
        nuevos += 1
    huerfanos = [k for k in have if k not in ids_now]
    for k in huerfanos:
        con.execute("DELETE FROM pedido_embeddings WHERE pedido_id=?", (k,))
    con.commit()
    print(f"preindex pedidos: {nuevos} (re)embebidos, {saltados} sin cambios, "
          f"{fallidos} fallidos, {len(huerfanos)} huerfanos eliminados")


def cmd_psearch(args):
    con = connect()
    qvec = embed(args.consulta, "query")
    where, params = "", []
    if args.estado:
        estados = [e.strip() for e in args.estado.split(",") if e.strip()]
        if estados:
            where = "WHERE p.estado IN (%s)" % ",".join("?" * len(estados))
            params = estados
    rows = con.execute(
        f"SELECT p.id,p.estado,p.prioridad,p.titulo,COALESCE(p.vence,''),e.vec "
        f"FROM mem.pedidos p JOIN pedido_embeddings e ON e.pedido_id=p.id {where}",
        params).fetchall()
    scored = []
    for pid, estado, prio, titulo, vence, vec in rows:
        v = array.array("f")
        v.frombytes(vec)
        if len(v) != len(qvec):
            continue
        scored.append((cosine(qvec, v), pid, estado, prio, titulo, vence))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[: args.k]
    if args.json:
        print(json.dumps(
            [{"score": round(s, 4), "id": i, "estado": e, "prioridad": p,
              "titulo": t, "vence": v} for s, i, e, p, t, v in top],
            ensure_ascii=False))
        return
    if not top:
        print("(sin resultados — ¿hay pedidos? ¿corriste 'mem preindex'?)")
        return
    for s, pid, estado, prio, titulo, vence in top:
        suf = f" (vence {vence})" if vence else ""
        print(f"[{s:.3f}] #{pid} [{estado}/{prio}] {titulo}{suf}")


# ---------- clustering de temas (areas y huecos) ----------
def cmd_clusters(args):
    """Agrupa memorias por similitud (componentes conexas con coseno >= umbral)
    para ver tus 'areas' reales. Las memorias sin vecino cercano salen como
    'aisladas' (posibles huecos de documentacion o notas sueltas). Puro embeddings;
    con --label el modelo instruct le pone un tema a cada grupo (best-effort)."""
    con = connect()
    where, params = "WHERE m.activo=1", []
    if args.device:
        where += " AND m.dispositivo=?"
        params.append(args.device)
    elif args.here:
        where += " AND (m.dispositivo='compartida' OR m.dispositivo=?)"
        params.append(args.here_device)
    rows = con.execute(
        f"SELECT m.tipo,m.nombre,m.resumen,e.vec "
        f"FROM mem.memorias m JOIN embeddings e ON e.memoria_id=m.id {where}",
        params).fetchall()
    items = []
    for tipo, nombre, resumen, vec in rows:
        v = array.array("f")
        v.frombytes(vec)
        items.append((tipo, nombre, resumen, v))
    n = len(items)
    if n == 0:
        print("(sin memorias indexadas — ¿corriste 'mem reindex'?)")
        return
    # union-find sobre pares con coseno >= umbral
    parent = list(range(n))

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for i in range(n):
        vi = items[i][3]
        for j in range(i + 1, n):
            vj = items[j][3]
            if len(vi) != len(vj):
                continue
            if cosine(vi, vj) >= args.threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = sorted(groups.values(), key=len, reverse=True)
    multi = [c for c in clusters if len(c) >= args.min_size]
    singles = [i for c in clusters if len(c) < args.min_size for i in c]

    def label_for(idxs):
        if not args.label:
            return ""
        resumenes = "\n".join(f"- {items[i][2]}" for i in idxs[:8])
        try:
            out = gen_json(
                "Eres un bibliotecario. Dado un conjunto de notas personales, "
                "identifica el tema comun en pocas palabras. Responde SOLO JSON: "
                '{"tema":"<2 a 4 palabras>"}',
                f"Notas:\n{resumenes}\n\n¿Cual es el tema comun?")
            return str(out.get("tema", "")).replace("\n", " ").strip()
        except Exception:
            return ""  # etiqueta best-effort: nunca debe tumbar el reporte (ni el JSON)

    if args.json:
        print(json.dumps({
            "umbral": args.threshold,
            "grupos": [{"tema": label_for(c),
                        "miembros": [{"tipo": items[i][0], "nombre": items[i][1]} for i in c]}
                       for c in multi],
            "aisladas": [{"tipo": items[i][0], "nombre": items[i][1]} for i in singles],
        }, ensure_ascii=False))
        return
    print(f"# Temas (umbral {args.threshold}, {n} memorias, {len(multi)} grupos, "
          f"{len(singles)} aisladas)")
    for k, c in enumerate(multi, 1):
        lab = label_for(c)
        print(f"## Grupo {k} ({len(c)})" + (f" — {lab}" if lab else ""))
        for i in sorted(c, key=lambda i: (items[i][0], items[i][1])):
            print(f"   [{items[i][0]}] {items[i][1]}")
    if singles:
        print(f"## Aisladas ({len(singles)}) — sin vecino >= {args.threshold} "
              f"(posibles huecos o notas sueltas)")
        for i in sorted(singles, key=lambda i: (items[i][0], items[i][1])):
            print(f"   [{items[i][0]}] {items[i][1]}")


# ---------- redaccion asistida: tags y resumen via modelo (#6/#7) ----------
def _norm_tag(t):
    # Translitera acentos (configuración -> configuracion) en vez de borrarlos,
    # asi los tags en español no se mutilan ni divergen del vocabulario existente.
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode()
    t = t.strip().lower().replace(" ", "-")
    return re.sub(r"[^a-z0-9_-]", "", t)


def _tag_vocab(con, top=60):
    """Tags existentes ordenados por frecuencia: vocabulario a REUSAR (consistencia)."""
    freq = {}
    for (tags,) in con.execute(
            "SELECT tags FROM mem.memorias WHERE tags IS NOT NULL AND tags<>''"):
        for t in tags.split(","):
            nt = _norm_tag(t)
            if nt:
                freq[nt] = freq.get(nt, 0) + 1
    return [t for t, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top]]


def suggest_tags(texto, vocab):
    """Tags consistentes (reusa vocab). Lanza GenError si el modelo falla."""
    sysmsg = (
        "Eres un archivista que pone etiquetas consistentes a notas personales. "
        "REUSA etiquetas del vocabulario existente siempre que apliquen; crea "
        "nuevas SOLO si es imprescindible. Espanol neutro. Responde SOLO JSON: "
        '{"tags":["tag1","tag2"]} con 3 a 6 tags en minusculas, sin espacios '
        "(usa guiones).")
    user = (f"Vocabulario existente: {', '.join(vocab) or '(vacio)'}\n\n"
            f"Nota:\n{texto[:1500]}\n\n¿Que tags le pones?")
    out = gen_json(sysmsg, user)
    raw = out.get("tags", [])
    if not isinstance(raw, list):
        return []
    seen, tags = set(), []
    for t in raw:
        if not isinstance(t, str):
            continue  # descarta null/numero/dict que el modelo cuele en el array
        nt = _norm_tag(t)
        if nt and nt not in ("none", "null") and nt not in seen:
            seen.add(nt)
            tags.append(nt)
    return tags[:6]


def suggest_resumen(texto):
    """Resumen de una frase. Lanza GenError si el modelo falla; '' si es invalido."""
    sysmsg = (
        "Resume la nota en UNA frase concisa (max ~160 caracteres), espanol neutro "
        "(tuteo, sin voseo ni regionalismos). Responde SOLO JSON: {\"resumen\":\"...\"}.")
    out = gen_json(sysmsg, texto[:3000])
    r = out.get("resumen", "")
    if not isinstance(r, str):
        return ""  # null/lista/numero -> tratamos como 'sin sugerencia'
    return r.replace("\n", " ").replace("\t", " ").replace("\r", " ").strip()[:200]


def _subject_text(con, args):
    """Texto objetivo: candidata (--contenido / --resumen sin --nombre) o memoria
    existente (lookup por --nombre)."""
    if args.contenido or (args.resumen and not args.nombre):
        return doc_text(args.tipo, args.nombre, args.resumen, args.contenido, "")
    row = con.execute(
        "SELECT tipo,nombre,resumen,contenido FROM mem.memorias "
        "WHERE nombre=?", (args.nombre,)).fetchone()  # sin activo=1: retag/summarize valen para inactivas
    if not row:
        return ""
    tipo, nombre, resumen, contenido = row
    return doc_text(tipo, nombre, resumen, contenido, "")


def cmd_suggest_tags(args):
    con = connect()
    texto = _subject_text(con, args)
    if not texto.strip():
        return
    try:
        print(",".join(suggest_tags(texto, _tag_vocab(con))))
    except GenError:
        return


def cmd_suggest_resumen(args):
    con = connect()
    texto = _subject_text(con, args)
    if not texto.strip():
        return
    try:
        print(suggest_resumen(texto))
    except GenError:
        return


def cmd_dupes(args):
    """Memorias existentes semanticamente similares a una candidata.

    Pensado para 'mem remember': antes de insertar, avisa si ya hay algo
    parecido con OTRO nombre (los nombres iguales son un update via UPSERT,
    no un duplicado). Compara documento-vs-documento (mismo rol que el indice).

    Salida (sin --json): una linea TSV por match, ordenadas por score desc:
        score \\t tipo \\t nombre \\t dispositivo \\t resumen
    Vacia si no hay nada por encima del umbral.

    No fatal: si Ollama esta caido salimos en silencio (exit 0, sin salida)
    para no bloquear el guardado por no poder deduplicar.
    """
    text = doc_text(args.tipo, args.nombre, args.resumen, args.contenido, args.tags)
    if not text.strip():
        return
    try:
        qvec = embed(text, "document")
    except (SystemExit, EmbedError):
        return  # Ollama caido o item no-embebible: no deduplicamos, dejamos guardar.
    con = connect()
    rows = con.execute(
        "SELECT m.tipo,m.nombre,m.resumen,m.dispositivo,e.vec "
        "FROM mem.memorias m JOIN embeddings e ON e.memoria_id=m.id "
        "WHERE m.activo=1 AND m.nombre<>?",
        (args.exclude_nombre or args.nombre,),
    ).fetchall()
    scored = []
    for tipo, nombre, resumen, disp, vec in rows:
        v = array.array("f")
        v.frombytes(vec)
        if len(v) != len(qvec):
            continue  # otro modelo/dim: no comparable (largo real, no la columna dim)
        s = cosine(qvec, v)
        if s >= args.threshold:
            scored.append((s, tipo, nombre, resumen, disp))
    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[: args.k]
    if args.json:
        print(json.dumps(
            [{"score": round(s, 4), "tipo": t, "nombre": n, "resumen": r, "dispositivo": d}
             for s, t, n, r, d in scored], ensure_ascii=False))
        return
    # El consumidor (bash) parte por TAB y espera 1 linea fisica por match:
    # neutralizamos tabs/saltos en los campos para no romper el split.
    def flat(x):
        return (x or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
    for s, tipo, nombre, resumen, disp in scored:
        print(f"{s:.3f}\t{flat(tipo)}\t{flat(nombre)}\t{flat(disp)}\t{flat(resumen)}")


def cmd_conflicts(args):
    """Contradicciones entre una memoria (o candidata) y sus relacionadas.

    Embeddings acotan el conjunto (mismo tema); el modelo instruct juzga si hay
    choque real. Mejor-esfuerzo en dos sentidos: si no hay relacionadas NO se
    invoca al modelo (rapido); si el modelo no esta disponible, sale en silencio
    sin bloquear nada.

    Salida TSV (sin --json): score \\t nombre \\t confianza \\t motivo (solo choques).
    """
    def _empty():
        if args.json:
            print("[]")  # JSON siempre valido, distinga o no 'sin relacionadas'
    con = connect()
    if args.resumen:  # modo candidata (p.ej. desde 'remember')
        tipo, nombre = args.tipo, args.nombre
        resumen, contenido, tags = args.resumen, args.contenido, args.tags
    else:             # modo memoria existente (por nombre)
        row = con.execute(
            "SELECT tipo,nombre,resumen,contenido,COALESCE(tags,'') "
            "FROM mem.memorias WHERE nombre=? AND activo=1", (args.nombre,)).fetchone()
        if not row:
            return _empty()
        tipo, nombre, resumen, contenido, tags = row
    text = doc_text(tipo, nombre, resumen, contenido, tags)
    if not text.strip():
        return _empty()
    try:
        qvec = embed(text, "document")
    except (SystemExit, EmbedError):
        return _empty()
    rel = related_memories(con, qvec, nombre, args.rel, args.k)
    if not rel:
        return _empty()  # nada relacionado -> no se invoca el modelo generativo
    a_text = _judge_text(resumen, contenido)
    conflicts = []
    for s, rtipo, rnombre, rresumen, rcontenido, rdisp in rel:
        try:
            ver, conf, motivo = judge_contradiction(
                nombre, a_text, rnombre, _judge_text(rresumen, rcontenido))
        except GenError:
            # modelo no disponible/timeout: avisamos (a stderr) y devolvemos lo
            # acumulado para no presentar un chequeo truncado como "sin choques".
            print("(modelo no disponible: chequeo de contradicciones incompleto)",
                  file=sys.stderr)
            break
        if ver == "contradice" and conf >= args.min_confianza:
            conflicts.append((s, rnombre, conf, motivo))
    if args.json:
        print(json.dumps(
            [{"score": round(s, 4), "nombre": n, "confianza": round(c, 2), "motivo": m}
             for s, n, c, m in conflicts], ensure_ascii=False))
        return
    for s, n, c, m in conflicts:
        print(f"{s:.3f}\t{n}\t{c:.2f}\t{m}")


_VERSION_RE = re.compile(
    r"\b(?:v\d+\.\d+(?:\.\d+)?|\d+\.\d+\.\d+(?:[-.][0-9A-Za-z]+)*|(?:beta|alpha|rc)[.\-]?\d+)\b",
    re.I)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# Una IPv4 (cuatro octetos) NO es una version: la excluimos del flag de version.
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def _valid_date(s):
    """True si 's' es una fecha de calendario real (descarta 2026-13-45, etc.)."""
    try:
        _dt.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _stale_scan(con, days, date_age, flag_versions=False):
    """Devuelve (hoy, cutoff, flagged). flagged: (dias, fecha, tipo, nombre, disp, motivos).

    Dispara SOLO por antigüedad (sin tocar >= days): una fecha o versión escrita
    en el texto NO implica que la memoria esté obsoleta (p.ej. 'desde 2020 usamos
    X' sigue vigente). Las fechas viejas (validadas) y las versiones solo
    ENRIQUECEN un item ya marcado por edad; con flag_versions además se marcan
    los items con versiones fijadas (revisión manual).
    """
    hoy = con.execute("SELECT date('now','localtime')").fetchone()[0]
    cutoff = con.execute(
        "SELECT date('now','localtime',?)", (f"-{date_age} days",)).fetchone()[0]
    rows = con.execute(
        "SELECT nombre,tipo,dispositivo,resumen,contenido,"
        " CAST(julianday('now','localtime')"
        "      -julianday(COALESCE(actualizado,'1970-01-01')) AS INT) AS dias,"
        " substr(COALESCE(actualizado,''),1,10) "
        "FROM mem.memorias WHERE activo=1 ORDER BY dias DESC").fetchall()
    flagged = []
    for nombre, tipo, disp, resumen, contenido, dias, fecha in rows:
        dias = dias or 0  # actualizado NULL/ilegible -> julianday NULL -> None
        blob = f"{resumen or ''}\n{contenido or ''}"
        vers = [v for v in dict.fromkeys(_VERSION_RE.findall(blob))
                if any(c.isdigit() for c in v) and not _IPV4_RE.match(v)]
        motivos = []
        if dias >= days:
            motivos.append(f"sin tocar hace {dias}d")
            viejas = sorted({d for d in _DATE_RE.findall(blob)
                             if _valid_date(d) and d < cutoff})
            if viejas:
                motivos.append("fechas: " + ", ".join(viejas[:3]))
            if vers:
                motivos.append("versiones: " + ", ".join(vers[:3]))
        elif flag_versions and vers:
            motivos.append("versión fijada: " + ", ".join(vers[:3]))
        if motivos:
            flagged.append((dias, fecha, tipo, nombre, disp, "; ".join(motivos)))
    return hoy, cutoff, flagged


def cmd_stale(args):
    con = connect()
    hoy, cutoff, flagged = _stale_scan(con, args.days, args.date_age, args.versions)
    if args.json:
        print(json.dumps(
            [{"dias": d, "fecha": f, "tipo": t, "nombre": n, "dispositivo": dp, "motivos": m}
             for d, f, t, n, dp, m in flagged], ensure_ascii=False))
        return
    if not flagged:
        print(f"sin memorias marcadas (umbral {args.days}d, fechas < {cutoff})")
        return
    print(f"# Staleness — {len(flagged)} marcadas (hoy {hoy}, umbral {args.days}d)")
    for d, f, t, n, dp, m in flagged:
        print(f"  {d:>4}d  [{t}] {n} ({dp}) — {m}")


def cmd_audit(args):
    """Panel: staleness (sin modelo) + contradicciones entre pares relacionados."""
    con = connect()
    hoy, cutoff, flagged = _stale_scan(con, args.days, args.date_age, args.versions)
    if not flagged:
        print(f"# Staleness — sin marcadas (umbral {args.days}d)")
    else:
        print(f"# Staleness — {len(flagged)} marcadas (hoy {hoy}, umbral {args.days}d)")
        for d, f, t, n, dp, m in flagged:
            print(f"  {d:>4}d  [{t}] {n} ({dp}) — {m}")
    print()
    rows = con.execute(
        "SELECT m.tipo,m.nombre,m.resumen,m.contenido,e.vec "
        "FROM mem.memorias m JOIN embeddings e ON e.memoria_id=m.id "
        "WHERE m.activo=1").fetchall()
    items = []
    for tipo, nombre, resumen, contenido, vec in rows:
        v = array.array("f")
        v.frombytes(vec)
        items.append((tipo, nombre, resumen, contenido, v))
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if len(items[i][4]) != len(items[j][4]):
                continue
            s = cosine(items[i][4], items[j][4])
            if s >= args.rel:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    capped = pairs[: args.max_pairs]
    extra = f" (de {len(pairs)}, cap {args.max_pairs})" if len(pairs) > len(capped) else ""
    print(f"# Contradicciones — evaluando {len(capped)} pares relacionados (sim>={args.rel}){extra}")
    found = 0
    for idx, (s, i, j) in enumerate(capped, 1):
        a, b = items[i], items[j]
        print(f"  [{idx}/{len(capped)}] {a[1]} ~ {b[1]}", file=sys.stderr)
        try:
            ver, conf, motivo = judge_contradiction(
                a[1], _judge_text(a[2], a[3]), b[1], _judge_text(b[2], b[3]))
        except GenError as e:
            print(f"  (modelo no disponible: {e}) — se omite el resto")
            break
        if ver == "contradice" and conf >= args.min_confianza:
            found += 1
            print(f"  ⚠ [{conf:.2f}] {a[1]}  ⨯  {b[1]} (sim {s:.2f})")
            print(f"        {motivo}")
    if found == 0:
        print("  sin contradicciones detectadas")


# ---------- RAG sobre archivos del disco ----------
_HOME = os.path.expanduser("~")
# Directorios que nunca aportan (ruido/binarios/caches) o que guardan secretos.
# NO se descartan TODOS los ocultos: dotfiles y ~/.config son indexables.
_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".cache", ".venv",
    "venv", ".tox", "target", "dist", "build", ".next", "out", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "vendor", ".cargo", ".rustup", ".npm",
    ".gradle", ".idea", ".m2", ".steam", "Steam", "Trash",
    # secretos / credenciales: nunca indexar por defecto
    ".ssh", ".gnupg", ".aws", ".password-store", ".docker", ".kube",
    ".mozilla", "keyrings", "gcloud", "Bitwarden",
}
_IGNORE_EXT = {
    ".lock", ".map", ".pyc", ".pyo", ".o", ".a", ".so", ".dll", ".dylib",
    ".bin", ".class", ".jar", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".pdf", ".zip", ".gz", ".xz", ".zst", ".tar", ".7z", ".mp3", ".mp4", ".mkv",
    ".webm", ".wav", ".flac", ".ttf", ".otf", ".woff", ".woff2", ".sqlite", ".db",
}
# --- filtro de secretos a nivel de archivo (basename / extension / contenido) ---
_SECRET_EXT = {".pem", ".key", ".pfx", ".p12", ".kdbx", ".gpg", ".asc",
               ".jks", ".keystore", ".ppk"}
_SECRET_NAMES = {".netrc", ".pgpass", ".npmrc", ".git-credentials", ".htpasswd",
                 ".pypirc", ".dockercfg", ".bash_history", ".zsh_history",
                 ".python_history", "credentials", "secrets", "secret"}
_SECRET_KEYNAME = re.compile(r"^id_(rsa|ed25519|ecdsa|dsa)$")  # clave privada (sin .pub)
_SECRET_MARKERS = (b"PRIVATE KEY-----", b"OPENSSH PRIVATE KEY",
                   b"BEGIN PGP PRIVATE", b"aws_secret_access_key")


def _looks_secret(path, head):
    """Heuristica: ¿el archivo parece contener secretos? (no indexar por defecto)."""
    base = os.path.basename(path)
    low = base.lower()
    if base in _SECRET_NAMES or low.startswith(".env"):
        return True
    if os.path.splitext(low)[1] in _SECRET_EXT:
        return True
    if _SECRET_KEYNAME.match(base):
        return True
    return any(mark in head for mark in _SECRET_MARKERS)


def _short(p):
    return ("~" + p[len(_HOME):]) if p.startswith(_HOME) else p


def _snippet(t, n=140):
    for ln in t.splitlines():
        ln = ln.strip()
        if ln:
            return ln[:n]
    return t[:n].strip()


def _walk_files(root):
    if os.path.isfile(root):
        yield root
        return

    def _onerr(e):
        print(f"  sin acceso: {getattr(e, 'filename', root)}: {e}", file=sys.stderr)

    for dp, dns, fns in os.walk(root, onerror=_onerr):
        dns[:] = [d for d in dns if d not in _IGNORE_DIRS]
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in _IGNORE_EXT:
                continue
            yield os.path.join(dp, fn)


def _is_text_file(path, max_bytes=1_000_000):
    """True si es texto utf-8 razonable, no vacio y bajo el limite de tamaño."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    if st.st_size == 0 or st.st_size > max_bytes:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return False
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _chunk_lines(text, size=1500):
    """Trozos de ~size chars por lineas completas; cada uno con su linea de inicio.
    Una linea gigante (minified/base64) se parte por caracteres para no generar
    un chunk enorme (mantiene el chunk <= ~size, asi el texto guardado coincide
    con lo embebido)."""
    chunks, cur, cur_len, start = [], [], 0, 1
    for i, ln in enumerate(text.splitlines(), 1):
        if len(ln) > size:
            if cur:
                chunks.append((start, "\n".join(cur)))
                cur, cur_len = [], 0
            for off in range(0, len(ln), size):
                chunks.append((i, ln[off:off + size]))
            start = i + 1
            continue
        if cur and cur_len + len(ln) + 1 > size:
            chunks.append((start, "\n".join(cur)))
            cur, cur_len, start = [], 0, i
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        chunks.append((start, "\n".join(cur)))
    return chunks


def _normalize(v):
    """Normaliza el vector a norma 1 in-place (para que la busqueda use producto
    punto = coseno, mas barato que recomputar normas en cada consulta)."""
    n = sum(x * x for x in v) ** 0.5
    if n:
        for i in range(len(v)):
            v[i] = v[i] / n
    return v


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def cmd_index(args):
    """Indexa archivos (apuntes/configs/dotfiles) para busqueda semantica.

    Incremental: salta archivos sin cambios (atajo por mtime, hash como verdad;
    el hash incluye el MODELO, asi cambiar de embedder fuerza reindexado). Sin
    paths, refresca las raices ya indexadas. Por defecto NO indexa archivos que
    parecen secretos (claves, .env, history...): usar --include-secrets para forzar.
    """
    con = connect()
    if args.paths:
        roots = [os.path.realpath(p) for p in args.paths]
    else:
        roots = [r[0] for r in con.execute("SELECT ruta FROM archivos_roots")]
    if not roots:
        sys.exit("uso: mem index <ruta>...  (la 1a vez indica que indexar)")
    for r in roots:
        con.execute("INSERT OR IGNORE INTO archivos_roots(ruta) VALUES(?)", (r,))
    con.commit()
    # have: ruta_real -> (hash, mtime). hash incluye MODEL.
    have = {}
    for ruta, hsh, mt in con.execute("SELECT DISTINCT ruta,hash,mtime FROM archivos"):
        have[ruta] = (hsh, mt)
    size = max(200, min(args.chunk_size, 4000))
    n_idx = n_skip = n_chunks = n_secret = n_partial = 0
    seen = set()
    for root in roots:
        for path in _walk_files(root):
            rp = os.path.realpath(path)
            if rp in seen:          # raices solapadas o symlink->target ya visto
                continue
            seen.add(rp)
            try:
                mtime = os.path.getmtime(rp)
            except OSError:
                continue
            prev = have.get(rp)
            # atajo: mtime sin cambios y no quedo parcial -> sin releer ni hashear
            if prev and prev[1] == mtime and not str(prev[0]).endswith("~partial"):
                n_skip += 1
                continue
            if not _is_text_file(rp):
                continue
            try:
                with open(rp, "rb") as f:
                    head = f.read(4096)
            except OSError:
                continue
            if not args.include_secrets and _looks_secret(rp, head):
                n_secret += 1
                continue
            try:
                data = open(rp, "rb").read()
            except OSError:
                continue
            if b"\x00" in data:     # cola binaria que paso el chequeo de cabecera
                continue
            hh = hashlib.sha256(data)
            hh.update(MODEL.encode())
            h = hh.hexdigest()
            if prev and prev[0] == h:   # contenido+modelo identicos pese al mtime
                n_skip += 1
                continue
            text = data.decode("utf-8", errors="replace")
            # Embebemos TODOS los chunks ANTES de tocar la DB: si Ollama se cae a
            # mitad, las filas viejas quedan intactas (sin estado a medias).
            vecs, failed = [], False
            for idx, (start, ctext) in enumerate(_chunk_lines(text, size)):
                if not ctext.strip():
                    continue
                try:
                    v = embed(ctext, "document")
                except EmbedError as e:
                    failed = True
                    print(f"  saltado {_short(rp)}#{idx}: {e}", file=sys.stderr)
                    continue
                except SystemExit:
                    con.commit()
                    print(f"index abortado (Ollama caido): {n_idx} archivos hechos, "
                          f"'{_short(rp)}' sin completar; reintenta luego",
                          file=sys.stderr)
                    sys.exit(1)
                vecs.append((idx, start, ctext, _normalize(v)))
            # hash '~partial' si fallo algun chunk: el proximo run lo reintenta
            store_hash = h if not failed else h + "~partial"
            if failed:
                n_partial += 1
            con.execute("DELETE FROM archivos WHERE ruta=?", (rp,))
            for idx, start, ctext, v in vecs:
                con.execute(
                    "INSERT INTO archivos"
                    "(ruta,chunk,inicio,texto,hash,modelo,dim,vec,mtime) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (rp, idx, start, ctext, store_hash, MODEL, len(v),
                     v.tobytes(), mtime))
            con.commit()
            have[rp] = (store_hash, mtime)   # refresca snapshot (evita doble-embed)
            n_idx += 1
            n_chunks += len(vecs)
            print(f"  indexado: {_short(rp)} ({len(vecs)} chunks)", file=sys.stderr)
    extra = f", {n_secret} sensibles omitidos" if n_secret else ""
    extra += f", {n_partial} parciales (reintentables)" if n_partial else ""
    print(f"index: {n_idx} (re)indexados, {n_skip} sin cambios, "
          f"{n_chunks} chunks{extra}; {len(roots)} raices")


def cmd_fsearch(args):
    con = connect()
    qvec = _normalize(embed(args.consulta, "query"))
    rows = con.execute("SELECT ruta,inicio,texto,vec FROM archivos").fetchall()
    if len(rows) > 20000:
        print(f"(aviso: {len(rows)} chunks indexados; la busqueda puede tardar "
              f"unos segundos)", file=sys.stderr)
    scored = []
    for ruta, inicio, texto, vec in rows:
        v = array.array("f")
        v.frombytes(vec)
        if len(v) != len(qvec):
            continue  # otro modelo/dim: no comparable
        scored.append((_dot(qvec, v), ruta, inicio, texto))  # vecs normalizados -> dot=coseno
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[: args.k]
    if args.json:
        print(json.dumps(
            [{"score": round(s, 4), "ruta": r, "linea": i, "snippet": _snippet(t)}
             for s, r, i, t in top], ensure_ascii=False))
        return
    if not top:
        print("(sin resultados — ¿corriste 'mem index <ruta>'?)")
        return
    for s, ruta, inicio, texto in top:
        print(f"[{s:.3f}] {_short(ruta)}:{inicio}")
        print(f"        {_snippet(texto)}")


def cmd_files(args):
    con = connect()
    rows = con.execute(
        "SELECT ruta, COUNT(*) FROM archivos GROUP BY ruta ORDER BY ruta").fetchall()
    if args.json:
        print(json.dumps([{"ruta": r, "chunks": c} for r, c in rows], ensure_ascii=False))
        return
    roots = [r[0] for r in con.execute("SELECT ruta FROM archivos_roots ORDER BY ruta")]
    print(f"# Raices indexadas ({len(roots)}):")
    for r in roots:
        print(f"    {_short(r)}")
    tot = sum(c for _, c in rows)
    print(f"# Archivos: {len(rows)} ({tot} chunks)")
    for r, c in rows[:60]:
        print(f"  {c:>4}  {_short(r)}")


def cmd_fprune(args):
    con = connect()
    rutas = [r[0] for r in con.execute("SELECT DISTINCT ruta FROM archivos")]
    gone = []
    for r in rutas:
        try:
            os.lstat(r)
        except FileNotFoundError:
            gone.append(r)        # solo borramos si DEFINITIVAMENTE no existe
        except OSError:
            pass                  # inaccesible (disco desmontado/permiso): conservar
    for r in gone:
        con.execute("DELETE FROM archivos WHERE ruta=?", (r,))
    con.commit()
    print(f"fprune: {len(gone)} archivos borrados del indice "
          f"({len(rutas) - len(gone)} siguen)")


def cmd_embed_one(args):
    v = embed(args.texto, "query")
    print(f"dim={len(v)} primeros5={list(v[:5])}")


def cmd_warmup(args):
    """Precarga el modelo en VRAM (keep_alive largo). Pensado para correr
    detached al iniciar sesion, asi la primera busqueda ya esta caliente."""
    embed("warmup", "query")
    print(f"modelo {MODEL} cargado (keep_alive={KEEP_ALIVE})")


def cmd_gen_ready(args):
    """Exit 0 si el modelo generativo YA esta residente en VRAM (via /api/ps).

    Lo usa 'remember' para decidir: si esta caliente, chequea contradicciones
    sincronicamente (rapido); si esta frio, no frena el guardado y precalienta.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=3) as r:
            data = json.load(r)
    except (OSError, ValueError):
        sys.exit(1)
    base = GEN_MODEL.split(":")[0]
    for m in data.get("models", []):
        name = m.get("name", "") or m.get("model", "")
        if name == GEN_MODEL or name.startswith(base):
            sys.exit(0)
    sys.exit(1)


def cmd_gen_warmup(args):
    """Carga el modelo generativo en VRAM (best-effort, detached). Para que el
    proximo chequeo de contradicciones lo encuentre caliente."""
    try:
        gen_json("Responde solo JSON.", 'Devuelve {"ok":true}')
    except GenError:
        pass


def main():
    ap = argparse.ArgumentParser(prog="mem_vec")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reindex")
    r.add_argument("--all", action="store_true", help="incluye memorias inactivas")
    r.set_defaults(func=cmd_reindex)

    s = sub.add_parser("search")
    s.add_argument("consulta")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--here", action="store_true", help="filtra compartida + este equipo")
    s.add_argument("--here-device", default="")
    s.add_argument("--device", default="")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    for _name, _fn in (("suggest-tags", cmd_suggest_tags),
                       ("suggest-resumen", cmd_suggest_resumen)):
        sp = sub.add_parser(_name)
        sp.add_argument("--tipo", default="")
        sp.add_argument("--nombre", default="")
        sp.add_argument("--resumen", default="")
        sp.add_argument("--contenido", default="")
        sp.set_defaults(func=_fn)

    cl = sub.add_parser("clusters")
    cl.add_argument("--threshold", type=float,
                    default=float(os.environ.get("MEM_CLUSTER_THRESHOLD", "0.72")))
    cl.add_argument("--min-size", type=int, default=2)
    cl.add_argument("--label", action="store_true")
    cl.add_argument("--device", default="")
    cl.add_argument("--here", action="store_true")
    cl.add_argument("--here-device", default="")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=cmd_clusters)

    pix = sub.add_parser("preindex")
    pix.set_defaults(func=cmd_preindex)

    ps = sub.add_parser("psearch")
    ps.add_argument("consulta")
    ps.add_argument("-k", type=int, default=5)
    ps.add_argument("--estado", default="")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_psearch)

    d = sub.add_parser("dupes")
    d.add_argument("--tipo", default="")
    d.add_argument("--nombre", default="")
    d.add_argument("--resumen", default="")
    d.add_argument("--contenido", default="")
    d.add_argument("--tags", default="")
    d.add_argument("--exclude-nombre", default="",
                   help="nombre a excluir (default: el propio --nombre)")
    d.add_argument("--threshold", type=float,
                   default=float(os.environ.get("MEM_DEDUPE_THRESHOLD", "0.80")))
    d.add_argument("-k", type=int, default=5)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_dupes)

    cf = sub.add_parser("conflicts")
    cf.add_argument("--tipo", default="")
    cf.add_argument("--nombre", default="")
    cf.add_argument("--resumen", default="")
    cf.add_argument("--contenido", default="")
    cf.add_argument("--tags", default="")
    cf.add_argument("--rel", type=float,
                    default=float(os.environ.get("MEM_CONFLICT_REL", "0.75")))
    cf.add_argument("-k", type=int, default=5)
    cf.add_argument("--min-confianza", type=float,
                    default=float(os.environ.get("MEM_CONFLICT_MIN_CONF", "0.6")))
    cf.add_argument("--json", action="store_true")
    cf.set_defaults(func=cmd_conflicts)

    st = sub.add_parser("stale")
    st.add_argument("--days", type=int,
                    default=int(os.environ.get("MEM_STALE_DAYS", "120")))
    st.add_argument("--date-age", type=int, default=180)
    st.add_argument("--versions", action="store_true",
                    help="marca también memorias con versiones fijadas (revisión)")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_stale)

    au = sub.add_parser("audit")
    au.add_argument("--days", type=int,
                    default=int(os.environ.get("MEM_STALE_DAYS", "120")))
    au.add_argument("--date-age", type=int, default=180)
    au.add_argument("--versions", action="store_true",
                    help="marca también memorias con versiones fijadas (revisión)")
    au.add_argument("--rel", type=float,
                    default=float(os.environ.get("MEM_CONFLICT_REL", "0.75")))
    au.add_argument("--min-confianza", type=float,
                    default=float(os.environ.get("MEM_CONFLICT_MIN_CONF", "0.6")))
    au.add_argument("--max-pairs", type=int, default=40)
    au.set_defaults(func=cmd_audit)

    e = sub.add_parser("embed-one")
    e.add_argument("texto")
    e.set_defaults(func=cmd_embed_one)

    gr = sub.add_parser("gen-ready")
    gr.set_defaults(func=cmd_gen_ready)

    gw = sub.add_parser("gen-warmup")
    gw.set_defaults(func=cmd_gen_warmup)

    ix = sub.add_parser("index")
    ix.add_argument("paths", nargs="*")
    ix.add_argument("--chunk-size", type=int, default=1500)
    ix.add_argument("--include-secrets", action="store_true",
                    help="indexa tambien archivos que parecen secretos (peligroso)")
    ix.set_defaults(func=cmd_index)

    fs = sub.add_parser("fsearch")
    fs.add_argument("consulta")
    fs.add_argument("-k", type=int, default=8)
    fs.add_argument("--json", action="store_true")
    fs.set_defaults(func=cmd_fsearch)

    fl = sub.add_parser("files")
    fl.add_argument("--json", action="store_true")
    fl.set_defaults(func=cmd_files)

    fp = sub.add_parser("fprune")
    fp.set_defaults(func=cmd_fprune)

    w = sub.add_parser("warmup")
    w.set_defaults(func=cmd_warmup)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
