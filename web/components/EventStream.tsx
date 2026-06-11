'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { KIND_STYLES, TRACE_PHASE_ACCENT } from '@/lib/events';
import type { LogEntry } from '@/lib/types';
import { JsonView } from './JsonView';

export function EventStream({ entries }: { entries: LogEntry[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  const [showTrace, setShowTrace] = useState(true);

  const traceCount = useMemo(() => entries.filter((e) => e.kind === 'TRACE').length, [entries]);
  const visible = useMemo(
    () => (showTrace ? entries : entries.filter((e) => e.kind !== 'TRACE')),
    [entries, showTrace],
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [visible]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-slate-800/80 px-4 py-2">
        <span className="text-xs uppercase tracking-wider text-slate-500">Events</span>
        <button
          onClick={() => setShowTrace((s) => !s)}
          className={`mono rounded px-2 py-0.5 text-[10px] font-semibold transition-colors ${
            showTrace
              ? 'bg-indigo-500/20 text-indigo-300'
              : 'bg-slate-800/60 text-slate-500 hover:text-slate-300'
          }`}
          title="Toggle deep-trace frames (what flows between the agents)"
        >
          {showTrace ? '◉' : '◯'} trace ({traceCount})
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        {visible.length === 0 ? (
          <div className="text-sm text-slate-500">No events yet. Say hi 👋</div>
        ) : (
          <ul className="flex flex-col gap-1">
            {visible.flatMap((e, i) => {
              const row =
                e.kind === 'TRACE' ? <TraceRow key={e.id} entry={e} /> : <LogRow key={e.id} entry={e} />;
              // a USER row starts a new turn — delimit it from the previous one
              return e.kind === 'USER' && i > 0
                ? [<TurnDivider key={`div-${e.id}`} turn={e.turn} />, row]
                : [row];
            })}
          </ul>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function TurnDivider({ turn }: { turn?: number }) {
  return (
    <li className="my-1.5 flex items-center gap-2 select-none" aria-hidden>
      <span className="h-px flex-1 bg-slate-700/60" />
      <span className="mono rounded-full bg-slate-800 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
        {turn ? `Turn ${turn}` : 'New turn'}
      </span>
      <span className="h-px flex-1 bg-slate-700/60" />
    </li>
  );
}

function LogRow({ entry }: { entry: LogEntry }) {
  const s = KIND_STYLES[entry.kind];
  const [open, setOpen] = useState(false);
  const hasData = !!entry.payload && Object.keys(entry.payload).length > 0;
  return (
    <li className="flex flex-col">
      <div className="flex items-start gap-2 text-sm leading-relaxed">
        <span className="mono shrink-0 pt-1 text-[10px] text-slate-600">{entry.ts}</span>
        <span className={`mono shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${s.bg} ${s.fg}`}>
          {s.label}
        </span>
        <span className="min-w-0 flex-1 whitespace-pre-wrap break-words text-slate-200">
          {entry.body}
          {hasData && (
            <button
              onClick={() => setOpen((o) => !o)}
              className="ml-2 align-baseline text-[10px] text-slate-500 underline hover:text-slate-300"
            >
              {open ? 'hide' : 'data'}
            </button>
          )}
        </span>
      </div>
      {open && hasData && (
        <div className="mt-1 overflow-x-auto pl-16 text-xs">
          <JsonView value={entry.payload} />
        </div>
      )}
    </li>
  );
}

function TraceRow({ entry }: { entry: LogEntry }) {
  const [open, setOpen] = useState(false);
  const accent = TRACE_PHASE_ACCENT[entry.phase ?? ''] ?? 'border-indigo-400/70';
  return (
    <li className={`rounded-md border-l-2 bg-slate-900/40 ${accent}`}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-2 px-2 py-1 text-left text-sm"
      >
        <span className="mono shrink-0 pt-1 text-[10px] text-slate-600">{entry.ts}</span>
        <span className="mono shrink-0 rounded bg-indigo-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-300">
          TRACE
        </span>
        <span className="mono shrink-0 pt-0.5 text-[10px] text-slate-500">{entry.phase}</span>
        <span className="min-w-0 flex-1 text-slate-300">{entry.title ?? entry.body}</span>
        <span className="shrink-0 pt-0.5 text-xs text-slate-600">{open ? '▾' : '▸'}</span>
      </button>
      {open && entry.payload && (
        <div className="overflow-x-auto px-3 pb-2 pl-9 text-xs">
          <JsonView value={entry.payload} />
        </div>
      )}
    </li>
  );
}
