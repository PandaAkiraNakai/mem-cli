# mem

CLI mínimo en bash sobre SQLite para **memorias** (hechos persistentes que quieres cargar al inicio de cada sesión) y **pedidos** (TODOs persistentes con estado y prioridad).

Pensado originalmente como capa de memoria local para [Claude Code](https://claude.com/claude-code) — el output de `mem context` se mete como contexto al iniciar cada sesión, así el asistente "recuerda" lo que le pediste antes sin depender de servicios externos. Funciona igual de bien para cualquier flujo donde quieras estado persistente accesible por CLI.

## Modelo

Dos tablas, cada una con su finalidad clara:

- **memorias** — Hechos estables que quieres que estén siempre disponibles. Tipados: `user`, `feedback`, `project`, `reference`, `nota`. Únicas por `nombre`. Activables / desactivables (`activo`). Con **scope por dispositivo** (`dispositivo`): `compartida` (default, las ven todos los equipos) o una etiqueta de máquina (`torre`, `laptop`, `mac`, …) que solo se inyecta en ese equipo.
- **pedidos** — TODOs con `estado` (pendiente / en_progreso / hecho / cancelado), `prioridad` (baja / normal / alta / urgente) y `vence` opcional. Historial de cambios via trigger.

## Instalación

```bash
# Requisitos: bash 4+, sqlite3
curl -fsSL https://raw.githubusercontent.com/PandaAkiraNakai/mem-cli/main/mem \
  -o ~/.local/bin/mem
chmod +x ~/.local/bin/mem
```

La primera vez que corre, `mem` crea la DB en
`${XDG_DATA_HOME:-~/.local/share}/mem/memory.db` con el schema completo (tablas, índices, triggers). Si quieres otro path:

```bash
export MEM_DB=~/proyectos/mi-vault/memoria.db
```

## Uso

### Pedidos

```bash
mem add "Refactor del módulo X" --prio alta --vence 2026-06-15
mem add "Revisar PR #42" --detalle "Falta cubrir el caso null en el handler"

mem ls                          # pendientes y en progreso
mem ls --estado todos           # incluye hechos y cancelados
mem ls --prio urgente

mem show 3                      # ver detalle del pedido #3
mem wip 3                       # marcar en progreso
mem do 3                        # marcar hecho (timestampea completado)
mem cancel 3

mem note 3 "Bloqueado por revisión de seguridad"
mem rm 3                        # borrar definitivamente
```

### Memorias

```bash
mem remember user perfil "Soy backend dev, Python primario" \
  --contenido "Trabajo full-stack pero el 80% del tiempo es Python/FastAPI"

mem remember feedback estilo-tests "Tests sin mocks de DB" \
  --contenido "Preferir testcontainers o sqlite en memoria. Razón: el incidente del Q3 con mocks divergentes" \
  --tags "testing,arquitectura"

mem remember project monorepo-X "Migrando frontend de webpack a vite, deadline 2026-Q3"

mem mem-ls                      # memorias activas
mem mem-ls --tipo feedback      # filtrar por tipo
mem mem-ls --all                # incluye inactivas

mem mem-show estilo-tests       # contenido completo
mem mem-off estilo-tests        # desactivar (queda en la DB)
mem mem-on  estilo-tests
mem mem-rm  estilo-tests        # borrado definitivo
```

### Memorias por dispositivo

Cuando varias máquinas comparten la misma DB (p. ej. sincronizada vía git), la columna `dispositivo` separa lo común de la config propia de cada equipo. Cada máquina declara su etiqueta una vez:

```bash
mem device            # muestra la etiqueta actual
mem device laptop     # la fija (escribe un archivo .device junto a la DB)
```

Precedencia de la etiqueta: `$MEM_DEVICE` > archivo `.device` > `hostname`. El `.device` es local: no lo sincronices (agrégalo al `.gitignore`).

```bash
# memoria compartida (default) — la ven todos los equipos
mem remember reference api-key-rotacion "Las API keys rotan cada 90 días"

# memoria específica de este equipo — solo se inyecta acá
mem remember reference monitores "Layout: 3 pantallas, central a 144Hz" --dispositivo laptop

mem mem-ls --here              # solo compartidas + las de este equipo
mem mem-ls --dispositivo mac   # solo las de un equipo
```

`mem context` filtra automáticamente: inyecta las `compartida` + las del equipo actual, nunca la config de otra máquina.

### Contexto (para inyectar en una sesión)

```bash
mem context
```

Devuelve markdown con los pedidos pendientes y las memorias activas, listo para concatenar en un system prompt, en un session hook de Claude Code, o en cualquier otro lugar donde quieras que tu asistente "arranque sabiendo".

Ejemplo de output:

```markdown
## Pedidos pendientes (SQLite /home/user/.local/share/mem/memory.db)
- #3 [alta] Refactor del módulo X (vence 2026-06-15)
- #4 [normal] Revisar PR #42

## Memorias SQLite activas
- [user] perfil — Soy backend dev, Python primario
- [feedback] estilo-tests — Tests sin mocks de DB

(Usa 'mem mem-show <nombre>' para el contenido completo, 'mem ls' para los pedidos.)
```

### SQL crudo

```bash
mem raw "SELECT tipo, COUNT(*) FROM memorias WHERE activo=1 GROUP BY tipo;"
mem raw ".schema"
```

## Schema

```sql
CREATE TABLE memorias (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  tipo        TEXT NOT NULL CHECK(tipo IN ('user','feedback','project','reference','nota')),
  nombre      TEXT NOT NULL UNIQUE,
  resumen     TEXT NOT NULL,
  contenido   TEXT NOT NULL,
  tags        TEXT,
  dispositivo TEXT NOT NULL DEFAULT 'compartida',
  activo      INTEGER NOT NULL DEFAULT 1,
  creado      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  actualizado TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE pedidos (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  titulo      TEXT NOT NULL,
  detalle     TEXT,
  estado      TEXT NOT NULL DEFAULT 'pendiente'
              CHECK(estado IN ('pendiente','en_progreso','hecho','cancelado')),
  prioridad   TEXT NOT NULL DEFAULT 'normal'
              CHECK(prioridad IN ('baja','normal','alta','urgente')),
  vence       TEXT,
  notas       TEXT,
  creado      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  actualizado TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  completado  TEXT
);
```

Triggers automáticos: `actualizado` se refresca en cada UPDATE, y `completado` se timestampea cuando un pedido pasa a `hecho`.

## Capa semántica (opcional)

Si tienes muchas memorias, el listado plano de `mem context` deja de servir como contexto. La capa semántica indexa las memorias con un modelo de embeddings local y permite búsqueda por significado, dedupe al guardar, RAG sobre archivos del disco y búsqueda de tareas por tema.

### Dependencias

- [Ollama](https://ollama.com) corriendo local (CPU o GPU).
- Modelo de embeddings: `qwen3-embedding:4b` (multilingüe, ~2.5GB, recomendado).
- Opcional, para razonamiento (`mem conflicts`, etc.): `llama3.1:8b` instruct.
- Python 3 (sólo stdlib — no hay pip install).

```bash
ollama pull qwen3-embedding:4b
ollama pull llama3.1:8b           # opcional
```

El backend Python `mem_vec.py` se distribuye junto al script principal en este repo. Cópialo a `${XDG_DATA_HOME:-~/.local/share}/mem/mem_vec.py` (o configura `MEM_VEC_PY` apuntando a su ruta).

### Comandos

```bash
# busca por significado, no por substring
mem search "credenciales del servidor de prod"

# (re)indexa las memorias y pedidos activos
mem reindex

# precarga el modelo en GPU (keep_alive 30m, sobrevive al hook)
mem warmup

# busca tareas pendientes por tema
mem psearch "auth refactor"

# RAG sobre archivos del disco (apuntes, configs, dotfiles)
mem index ~/notas ~/.config/niri
mem fsearch "cómo configuré los binds de niri"
mem files                 # lista archivos indexados
mem fprune                # quita del índice los borrados

# detecta memorias en conflicto y obsoletas
mem conflicts
mem stale

# auditoría general y mantenimiento
mem audit
mem clusters              # agrupa memorias similares
mem retag                 # sugiere tags via LLM
```

### Modo lite en `mem context`

Por defecto `mem context` inyecta todas las memorias activas. Con muchas se desborda — el modo lite muestra sólo identidad + las reglas siempre activas y delega el resto a búsqueda on-demand:

```bash
mem context --lite
```

El hook de Claude Code recomendado es `mem context --lite` + `mem warmup` para no llenar el contexto al inicio y mantener el modelo caliente para los `mem search` que el asistente haga durante la sesión.

### Variables de entorno

| Variable | Default | Para qué |
|----------|---------|----------|
| `MEM_EMBED_MODEL` | `qwen3-embedding:4b` | Modelo de embeddings. Cambiar invalida el cache (rehash). |
| `MEM_GEN_MODEL` | `llama3.1:8b` | Modelo instruct para razonamiento. |
| `OLLAMA_URL` | `http://localhost:11434` | Endpoint de Ollama. |
| `MEM_KEEP_ALIVE` | `30m` | Tiempo que el modelo de embeddings queda en VRAM tras el último uso. |
| `MEM_DEDUPE_WARN` | `0.80` | Similitud sobre la que `mem remember` avisa de duplicado. |
| `MEM_DEDUPE_BLOCK` | `0.90` | Similitud sobre la que `mem remember` aborta (usa `--force` para saltarlo). |

Los embeddings van en un `vec.db` separado en el mismo directorio que `memory.db`. Es **datos derivados, regenerables** con `mem reindex` — agrégalo a tu `.gitignore` si sincronizas `memory.db` por git.

## Por qué bash + SQLite

- **Cero dependencias** más allá de `bash`, `sqlite3` y `coreutils`. Funciona en cualquier Linux y macOS sin instalar nada raro.
- **DB portable**: copiar el `.db` a otra máquina, listo. Backup con `cp`. Encriptado opcional con `sqlcipher` si quieres.
- **Migraciones**: idempotentes y automáticas. Al correr cualquier comando, `mem` agrega columnas faltantes (p. ej. `dispositivo`) sobre DBs viejas sin romper nada, así que actualizar el script es suficiente.

## Integración con Claude Code

Si usas [Claude Code](https://claude.com/claude-code), agregas un hook en `.claude/settings.json` que pegue `mem context` al inicio de cada sesión:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "mem context"
      }
    ]
  }
}
```

Cada conversación arranca con los pedidos pendientes y memorias activas ya en contexto.

## Licencia

MIT — ver [LICENSE](LICENSE).

<!-- profile-excerpt -->
CLI minimal en bash sobre **SQLite** para **memorias** (hechos persistentes inyectables al inicio de sesión) y **pedidos** (TODOs con estado y prioridad). Pensado como capa de memoria local para **Claude Code** vía hook `SessionStart`, pero sirve para cualquier flujo que necesite estado persistente accesible por CLI. Bootstrap automático del schema, XDG-compliant, cero dependencias más allá de `bash` + `sqlite3`.
<!-- /profile-excerpt -->
