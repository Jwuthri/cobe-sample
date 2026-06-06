'use client';

import { useState, KeyboardEvent } from 'react';

export function ChatInput({
  onSend,
  disabled,
}: {
  onSend: (msg: string) => void;
  disabled?: boolean;
}) {
  const [val, setVal] = useState('');

  function send() {
    const trimmed = val.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setVal('');
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex items-end gap-2 border-t border-slate-800/80 bg-slate-950/60 p-3">
      <textarea
        value={val}
        onChange={(e) => setVal(e.target.value)}
        onKeyDown={onKey}
        rows={2}
        placeholder={disabled ? 'Waiting for response…' : 'Type a message — Enter to send'}
        className="flex-1 resize-none rounded-md border border-slate-800/70 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-cyan-500/40 focus:ring-1 focus:ring-cyan-500/30 disabled:opacity-50"
        disabled={disabled}
      />
      <button
        onClick={send}
        disabled={disabled || !val.trim()}
        className="shrink-0 rounded-md bg-cyan-600 px-4 py-2 text-sm font-medium text-white shadow hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
      >
        Send
      </button>
    </div>
  );
}
