'use client';

import type { SessionInfo } from '@/lib/types';

export function Header({
  sessionId,
  sessions,
  readOnly,
  onNewSession,
  onLoadSession,
  onOpenSessions,
  busy,
}: {
  sessionId: string | null;
  sessions: SessionInfo[];
  readOnly: boolean;
  onNewSession: () => void;
  onLoadSession: (sid: string, live: boolean) => void;
  onOpenSessions: () => void;
  busy: boolean;
}) {
  const inList = sessions.some((s) => s.session_id === sessionId);

  return (
    <header className="flex items-center justify-between border-b border-slate-800/80 bg-slate-950/60 px-4 py-3">
      <div className="flex items-center gap-3">
        <div className="flex size-7 items-center justify-center rounded-md bg-cyan-500/20 text-cyan-300">
          <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="9" />
            <path d="M8 12h8M12 8v8" />
          </svg>
        </div>
        <div>
          <div className="text-sm font-semibold text-slate-100">agent_v2</div>
          <div className="text-[11px] uppercase tracking-wider text-slate-500">
            multi-agent debug console
          </div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        {/* Load a previous (persisted) session — replays its stored events. */}
        <select
          value={sessionId ?? ''}
          onFocus={onOpenSessions}
          onMouseDown={onOpenSessions}
          onChange={(e) => {
            const s = sessions.find((x) => x.session_id === e.target.value);
            if (s) onLoadSession(s.session_id, s.live);
          }}
          disabled={busy}
          title="Load a previous session"
          className="max-w-[230px] rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs text-slate-200 outline-none hover:border-slate-600 focus:border-cyan-500/40 disabled:opacity-50"
        >
          {!inList && sessionId && (
            <option value={sessionId}>{sessionId} · current</option>
          )}
          {sessions.length === 0 && (
            <option value="" disabled>
              No saved sessions yet
            </option>
          )}
          {sessions.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.session_id} · {s.turns} turn{s.turns === 1 ? '' : 's'}
              {s.live ? ' · live' : ''}
            </option>
          ))}
        </select>

        {sessionId && (
          <div className="text-right">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">
              session{readOnly ? ' · saved' : ''}
            </div>
            <div className="mono text-xs text-slate-300">{sessionId}</div>
          </div>
        )}
        <span
          className={`size-2 rounded-full ${
            busy
              ? 'animate-pulse bg-amber-400'
              : readOnly
                ? 'bg-amber-400'
                : sessionId
                  ? 'bg-emerald-400'
                  : 'bg-slate-600'
          }`}
          title={busy ? 'processing' : readOnly ? 'saved session (read-only)' : sessionId ? 'idle' : 'no session'}
        />
        <button
          onClick={onNewSession}
          className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:bg-slate-800"
        >
          New session
        </button>
      </div>
    </header>
  );
}
