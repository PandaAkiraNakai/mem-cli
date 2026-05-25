# mem

CLI mínimo en bash sobre SQLite para **memorias** (hechos persistentes que querés cargar al inicio de cada sesión) y **pedidos** (TODOs persistentes con estado y prioridad).

Pensado originalmente como capa de memoria local para [Claude Code](https://claude.com/claude-code) — el output de `mem context` se mete como contexto al iniciar cada sesión, así el asistente "recuerda" lo que le pediste antes sin depender de servicios externos. Funciona igual de bien para cualquier flujo donde quieras estado persistente accesible por CLI.

## Modelo

Dos tablas, cada una con su finalidad clara:

- **memorias** — Hechos estables que querés que estén siempre disponibles. Tipados: `user`, `feedback`, `project`, `reference`, `nota`. Únicas por `nombre`. Activables / desactivables (`activo`).
- **pedidos** — TODOs con `estado` (pendiente / en_progreso / hecho / cancelado), `prioridad` (baja / normal / alta / urgente) y `vence` opcional. Historial de cambios via trigger.

## Instalación

```bash
# Requisitos: bash 4+, sqlite3
curl -fsSL https://raw.githubusercontent.com/PandaAkiraNakai/mem-cli/main/mem \
  -o ~/.local/bin/mem
chmod +x ~/.local/bin/mem
```

La primera vez que corre, `mem` crea la DB en
`${XDG_DATA_HOME:-~/.local/share}/mem/memory.db` con el schema completo (tablas, índices, triggers). Si querés otro path:

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

## Por qué bash + SQLite

- **Cero dependencias** más allá de `bash`, `sqlite3` y `coreutils`. Funciona en cualquier Linux y macOS sin instalar nada raro.
- **DB portable**: copiar el `.db` a otra máquina, listo. Backup con `cp`. Encriptado opcional con `sqlcipher` si querés.
- **Migraciones**: hoy ninguna. Si el schema crece, se agregan `ALTER TABLE` idempotentes al bootstrap.

## Integración con Claude Code

Si usás [Claude Code](https://claude.com/claude-code), agregás un hook en `.claude/settings.json` que pegue `mem context` al inicio de cada sesión:

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
