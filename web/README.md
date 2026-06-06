# agent_v2 — debug console (Next.js)

A debugger UI for the multi-agent graph. Split layout:

- **Left**: live cart state (step indicator, items, blockers, totals)
- **Right**: events tab (USER → ROUTER → AGENT → SKILL → TOOL →
  STEP → WRITER → BOT, color-coded) and chat tab (bubble view)
- **Bottom**: chat input

Streams Server-Sent Events from the FastAPI bridge at `:8001`.

## Setup

```bash
# 1) backend (in another terminal)
cd python/agent_v2
pip install -e ".[server]"           # or: uv pip install -e ".[server]"
uvicorn server.main:app --reload --port 8001

# 2) frontend
cd web
npm install
npm run dev                          # http://localhost:3000
```

Next.js rewrites `/api/*` → `http://localhost:8001/api/*` (see
`next.config.mjs`). Point at a different backend with
`AGENT_V2_API_URL=https://… npm run dev`.

## What's in here

| File | Role |
|---|---|
| `app/page.tsx` | Main page: layout, session bootstrap, send-turn handler, SSE plumbing. |
| `app/layout.tsx` | Root `<html>` + global CSS import. |
| `app/globals.css` | Tailwind 4 + theme tokens + scrollbar polish. |
| `components/Header.tsx` | Top bar: session id, busy indicator, "New session" button. |
| `components/CartPanel.tsx` | The live state view. Step indicator, customer, address, items, totals, blockers. |
| `components/EventStream.tsx` | Color-coded log of every event. Auto-scrolls. |
| `components/ChatPanel.tsx` | Bubble view of the conversation (alternate tab). |
| `components/ChatInput.tsx` | Textarea + send button. Enter sends, Shift+Enter newline. |
| `lib/types.ts` | TypeScript shapes matching the FastAPI SSE payload. |
| `lib/api.ts` | `createSession`, `getState`, `streamTurn` (SSE parser). |
| `lib/events.ts` | Maps server events → `LogEntry` rows + per-kind color tokens. |

## How a turn flows in the UI

```
[user hits Enter]
   ↓
streamTurn(sessionId, msg, onEvent)
   ↓ POST /api/turn/<sid> → SSE
   ├─ {type:"user", content}     → optimistic chat bubble
   ├─ {type:"state", snapshot}   → CartPanel re-renders
   ├─ {type:"router", target}    → ROUTER log entry
   ├─ {type:"agent", node}       → AGENT log entry
   ├─ {type:"skill", name}       → SKILL log entry
   ├─ {type:"tool_start", ...}   → TOOL log entry
   ├─ {type:"tool_end", ...}     → RESULT log entry
   ├─ {type:"step", ...}         → STEP log entry
   ├─ {type:"writer", draft}     → pending bot bubble
   ├─ {type:"bot", content}      → final bot bubble
   └─ {type:"end"}               → unfreeze input
```

The page also re-fetches the authoritative state via `GET /api/state/<sid>`
on turn completion so any lost events don't desync the cart panel.

## Production notes

- Sessions are in-memory in the FastAPI process. Restart = clean slate.
  Swap `SESSIONS` for a checkpointer-backed store (SQLite/Postgres)
  for persistence.
- CORS is wide open in `server/main.py`. Tighten in production.
- The Next.js rewrite is the only "API client" — you can deploy the
  frontend behind any host that proxies `/api/*` to your FastAPI.
