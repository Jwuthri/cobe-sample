import type { AgentSnapshot, ServerEvent } from './types';

const API = ''; // same origin — Next.js rewrites /api/* → FastAPI

export async function createSession(): Promise<string> {
  const res = await fetch(`${API}/api/session`, { method: 'POST' });
  if (!res.ok) throw new Error(`create session failed: ${res.status}`);
  const json = (await res.json()) as { session_id: string };
  return json.session_id;
}

export async function getState(sessionId: string): Promise<AgentSnapshot> {
  const res = await fetch(`${API}/api/state/${sessionId}`);
  if (!res.ok) throw new Error(`get state failed: ${res.status}`);
  return (await res.json()) as AgentSnapshot;
}

/**
 * Run a turn and stream SSE events back. The callback is invoked once
 * per event, in order, until the server emits `{type:"end"}` or the
 * connection drops.
 */
export async function streamTurn(
  sessionId: string,
  message: string,
  onEvent: (ev: ServerEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API}/api/turn/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`turn failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // SSE frames are separated by `\n\n`. Each frame may have multiple lines.
    let idx = buf.indexOf('\n\n');
    while (idx !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = frame.split('\n').find((l) => l.startsWith('data:'));
      if (line) {
        const payload = line.slice(5).trim();
        if (payload) {
          try {
            onEvent(JSON.parse(payload) as ServerEvent);
          } catch (e) {
            console.error('parse SSE failed', payload, e);
          }
        }
      }
      idx = buf.indexOf('\n\n');
    }
  }
}
