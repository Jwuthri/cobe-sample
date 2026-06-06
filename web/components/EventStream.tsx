'use client';

import { useEffect, useRef } from 'react';
import { KIND_STYLES } from '@/lib/events';
import type { LogEntry } from '@/lib/types';

export function EventStream({ entries }: { entries: LogEntry[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [entries]);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-800/80 px-4 py-2 text-xs uppercase tracking-wider text-slate-500">
        Events
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        {entries.length === 0 ? (
          <div className="text-sm text-slate-500">No events yet. Say hi 👋</div>
        ) : (
          <ul className="flex flex-col gap-1">
            {entries.map((e) => {
              const s = KIND_STYLES[e.kind];
              return (
                <li key={e.id} className="flex items-start gap-2 text-sm leading-relaxed">
                  <span className="mono shrink-0 text-[10px] text-slate-600 pt-1">{e.ts}</span>
                  <span
                    className={`mono shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${s.bg} ${s.fg}`}
                  >
                    {s.label}
                  </span>
                  <span className="min-w-0 flex-1 whitespace-pre-wrap break-words text-slate-200">
                    {e.body}
                  </span>
                </li>
              );
            })}
          </ul>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
