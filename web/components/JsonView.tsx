'use client';

import { useState } from 'react';

// A compact, collapsible JSON tree for the trace panels. Objects/arrays are
// disclosure rows (auto-open near the top, collapsed deeper); long strings get a
// "+N more" expander so a giant prompt never floods the panel.

const STRING_PREVIEW = 360;
const AUTO_OPEN_DEPTH = 1; // objects/arrays at depth <= this start expanded

export function JsonView({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null) return <span className="text-slate-500">null</span>;
  if (value === undefined) return <span className="text-slate-500">—</span>;
  if (typeof value === 'string') return <JsonString value={value} />;
  if (typeof value === 'number') return <span className="text-amber-300">{value}</span>;
  if (typeof value === 'boolean') return <span className="text-purple-300">{String(value)}</span>;
  if (Array.isArray(value)) {
    if (looksLikeMessages(value)) return <MessageList messages={value as MsgLike[]} />;
    return <JsonBranch label={`[${value.length}]`} entries={value.map((v, i) => [String(i), v])} depth={depth} bracket="array" />;
  }
  if (typeof value === 'object')
    return (
      <JsonBranch
        label={`{${Object.keys(value as object).length}}`}
        entries={Object.entries(value as Record<string, unknown>)}
        depth={depth}
        bracket="object"
      />
    );
  return <span>{String(value)}</span>;
}

// ---- message arrays (conversation_seen / raw_messages) render as labeled rows ----
type MsgLike = {
  role?: unknown;
  content?: unknown;
  name?: unknown;
  note?: unknown;
  tool_calls?: unknown;
};

function looksLikeMessages(v: unknown[]): boolean {
  return (
    v.length > 0 &&
    v.every(
      (x) => x !== null && typeof x === 'object' && !Array.isArray(x) && 'role' in x && 'content' in x,
    )
  );
}

const ROLE_STYLE: Record<string, string> = {
  human: 'bg-sky-500/15 text-sky-300',
  ai: 'bg-violet-500/15 text-violet-300',
  system: 'bg-amber-500/15 text-amber-300',
  tool: 'bg-emerald-500/15 text-emerald-300',
};

function MessageList({ messages }: { messages: MsgLike[] }) {
  return (
    <ul className="flex flex-col gap-1">
      {messages.map((m, i) => {
        const role = String(m.role ?? '?');
        const chip = ROLE_STYLE[role] ?? 'bg-slate-700/40 text-slate-300';
        const toolCalls = Array.isArray(m.tool_calls) ? m.tool_calls : [];
        return (
          <li key={i} className="rounded bg-slate-950/50 px-2 py-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className={`mono rounded px-1.5 py-0.5 text-[10px] font-semibold ${chip}`}>{role}</span>
              {m.name ? <span className="mono text-[10px] text-slate-500">{String(m.name)}</span> : null}
              {m.note ? <span className="text-[10px] italic text-indigo-300/90">← {String(m.note)}</span> : null}
            </div>
            {m.content !== undefined && m.content !== '' ? (
              <div className="mt-0.5 whitespace-pre-wrap break-words text-emerald-200/90">
                <JsonString value={String(m.content)} />
              </div>
            ) : null}
            {toolCalls.length > 0 ? (
              <div className="mt-1 flex flex-col gap-0.5">
                {toolCalls.map((tc, j) => {
                  const call = tc as { name?: unknown; args?: unknown };
                  return (
                    <div key={j} className="mono text-[11px] text-amber-200/80">
                      ↳ {String(call.name)}({JSON.stringify(call.args ?? {})})
                    </div>
                  );
                })}
              </div>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

function JsonString({ value }: { value: string }) {
  const [open, setOpen] = useState(false);
  const long = value.length > STRING_PREVIEW;
  const text = !long || open ? value : value.slice(0, STRING_PREVIEW) + '…';
  return (
    <span className="whitespace-pre-wrap break-words text-emerald-200/90">
      {text}
      {long && (
        <button
          onClick={() => setOpen((o) => !o)}
          className="ml-1 align-baseline text-[10px] text-slate-500 underline hover:text-slate-300"
        >
          {open ? 'less' : `+${value.length - STRING_PREVIEW} more`}
        </button>
      )}
    </span>
  );
}

function JsonBranch({
  label,
  entries,
  depth,
  bracket,
}: {
  label: string;
  entries: [string, unknown][];
  depth: number;
  bracket: 'object' | 'array';
}) {
  const [open, setOpen] = useState(depth <= AUTO_OPEN_DEPTH);
  if (entries.length === 0) {
    return <span className="text-slate-500">{bracket === 'array' ? '[]' : '{}'}</span>;
  }
  return (
    <div className="inline-block w-full align-top">
      <button
        onClick={() => setOpen((o) => !o)}
        className="mono text-[10px] text-slate-500 hover:text-slate-300"
      >
        {open ? '▾' : '▸'} {label}
      </button>
      {open && (
        <ul className="ml-2 border-l border-slate-800 pl-3">
          {entries.map(([k, v]) => (
            <li key={k} className="py-0.5">
              <span className="mono text-sky-300/90">{k}</span>
              <span className="text-slate-600">: </span>
              <JsonView value={v} depth={depth + 1} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
