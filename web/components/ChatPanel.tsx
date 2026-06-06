'use client';

import { useEffect, useRef } from 'react';
import type { AgentSnapshot, Block } from '@/lib/types';
import { StructuredBlocks } from '@/components/blocks';

export function ChatPanel({
  snapshot,
  pendingUser,
  pendingBot,
}: {
  snapshot: AgentSnapshot | null;
  pendingUser: string | null;
  pendingBot: string | null;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const msgs = snapshot?.messages ?? [];

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [msgs.length, pendingUser, pendingBot]);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-800/80 px-4 py-2 text-xs uppercase tracking-wider text-slate-500">
        Conversation
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {msgs.length === 0 && !pendingUser ? (
          <div className="text-sm text-slate-500">No messages yet.</div>
        ) : (
          <ul className="flex flex-col gap-3">
            {msgs.map((m, i) => (
              <MessageRow key={i} role={m.role} content={m.content} blocks={m.blocks} />
            ))}
            {pendingUser && msgs[msgs.length - 1]?.content !== pendingUser && (
              <MessageRow role="human" content={pendingUser} />
            )}
            {pendingBot && <MessageRow role="ai" content={pendingBot} pending />}
          </ul>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

function MessageRow({
  role,
  content,
  blocks,
  pending = false,
}: {
  role: string;
  content: string;
  blocks?: Block[];
  pending?: boolean;
}) {
  const isUser = role === 'human' || role === 'user';
  return (
    <li className={`flex flex-col gap-1.5 ${isUser ? 'items-end' : 'items-start'}`}>
      <div
        className={`max-w-[80%] whitespace-pre-wrap break-words rounded-2xl px-3.5 py-2 text-sm leading-relaxed ${
          isUser
            ? 'bg-sky-500/20 text-sky-50 ring-1 ring-sky-400/30'
            : 'bg-slate-800/70 text-slate-100 ring-1 ring-slate-700/50'
        } ${pending ? 'opacity-70' : ''}`}
      >
        {content}
        {pending && <span className="ml-1 animate-pulse text-slate-400">▍</span>}
      </div>
      {!isUser && blocks && blocks.length > 0 && (
        <div className="w-full max-w-[90%]">
          <StructuredBlocks blocks={blocks} />
        </div>
      )}
    </li>
  );
}
