import type { LogEntry, ServerEvent } from './types';

let _counter = 0;
function id() {
  return `${Date.now()}-${_counter++}`;
}

function ts() {
  const d = new Date();
  return d.toTimeString().slice(0, 8); // HH:MM:SS
}

/**
 * Project a raw SSE event into 0..1 ``LogEntry`` rows for the
 * event-stream panel. ``state`` and ``end`` are control events that
 * don't render a log line (they update the cart panel / clear the
 * pending state instead).
 */
export function logEntriesFor(ev: ServerEvent): LogEntry[] {
  switch (ev.type) {
    case 'user':
      return [{ id: id(), ts: ts(), kind: 'USER', body: ev.content }];
    case 'router':
      return [
        {
          id: id(),
          ts: ts(),
          kind: 'ROUTER',
          body: `→ ${ev.target}${ev.iteration ? ` (iter ${ev.iteration})` : ''}`,
        },
      ];
    case 'agent':
      return [{ id: id(), ts: ts(), kind: 'AGENT', body: `${ev.node} finished` }];
    case 'skill':
      return [{ id: id(), ts: ts(), kind: 'SKILL', body: `load → ${ev.name}` }];
    case 'tool_start': {
      const args = Object.entries(ev.args || {})
        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
        .join(', ');
      return [{ id: id(), ts: ts(), kind: 'TOOL', body: `${ev.name}(${args})`, payload: ev.args }];
    }
    case 'tool_end': {
      const r = (ev.result || '').replace(/\n/g, ' ⏎ ');
      const short = r.length > 240 ? r.slice(0, 240) + '…' : r;
      return short ? [{ id: id(), ts: ts(), kind: 'TOOL_RESULT', body: `↳ ${short}` }] : [];
    }
    case 'step': {
      const parts = [`${ev.sop}: ${ev.summary}`];
      if (ev.asks.length) parts.push(`asks=[${ev.asks.join(', ')}]`);
      if (ev.next_sop) parts.push(`→ ${ev.next_sop}`);
      return [{ id: id(), ts: ts(), kind: 'STEP', body: parts.join(' '), payload: ev.details ?? undefined }];
    }
    case 'writer': {
      const draft = ev.draft.replace(/\n/g, ' ⏎ ');
      const short = draft.length > 200 ? draft.slice(0, 200) + '…' : draft;
      return [{ id: id(), ts: ts(), kind: 'WRITER', body: short }];
    }
    case 'gate':
      return ev.rejected
        ? [{ id: id(), ts: ts(), kind: 'GATE', body: `rejected: ${ev.errors.join('; ')}` }]
        : [];
    case 'validator':
      return [{ id: id(), ts: ts(), kind: 'VALIDATOR', body: ev.errors.join(', ') }];
    case 'bot':
      return [{ id: id(), ts: ts(), kind: 'BOT', body: ev.content }];
    case 'error':
      return [{ id: id(), ts: ts(), kind: 'ERROR', body: ev.content }];
    case 'end':
    case 'state':
      return [];
  }
}

export const KIND_STYLES: Record<LogEntry['kind'], { bg: string; fg: string; label: string }> = {
  USER: { bg: 'bg-sky-500/15', fg: 'text-sky-300', label: 'USER' },
  ROUTER: { bg: 'bg-blue-500/15', fg: 'text-blue-300', label: 'ROUTER' },
  AGENT: { bg: 'bg-violet-500/15', fg: 'text-violet-300', label: 'AGENT' },
  SKILL: { bg: 'bg-amber-500/15', fg: 'text-amber-300', label: 'SKILL' },
  TOOL: { bg: 'bg-emerald-500/15', fg: 'text-emerald-300', label: 'TOOL' },
  TOOL_RESULT: { bg: 'bg-emerald-500/5', fg: 'text-emerald-200/70', label: 'RESULT' },
  STEP: { bg: 'bg-orange-500/15', fg: 'text-orange-300', label: 'STEP' },
  WRITER: { bg: 'bg-teal-500/15', fg: 'text-teal-300', label: 'WRITER' },
  GATE: { bg: 'bg-red-500/15', fg: 'text-red-300', label: 'GATE' },
  VALIDATOR: { bg: 'bg-yellow-500/15', fg: 'text-yellow-300', label: 'VALIDATOR' },
  BOT: { bg: 'bg-cyan-500/15', fg: 'text-cyan-200', label: 'BOT' },
  ERROR: { bg: 'bg-rose-600/20', fg: 'text-rose-300', label: 'ERROR' },
};
