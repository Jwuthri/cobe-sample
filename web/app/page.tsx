'use client';

import { useCallback, useEffect, useState } from 'react';

import { CartPanel } from '@/components/CartPanel';
import { ChatInput } from '@/components/ChatInput';
import { ChatPanel } from '@/components/ChatPanel';
import { EventStream } from '@/components/EventStream';
import { Header } from '@/components/Header';
import { createSession, fetchEvents, getState, listSessions, streamTurn } from '@/lib/api';
import { formatStoredTs, logEntriesFor } from '@/lib/events';
import type { AgentSnapshot, LogEntry, SessionInfo } from '@/lib/types';

export default function Page() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<AgentSnapshot | null>(null);
  const [events, setEvents] = useState<LogEntry[]>([]);
  const [pendingUser, setPendingUser] = useState<string | null>(null);
  const [pendingBot, setPendingBot] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<'chat' | 'events'>('events');
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  // True while viewing an archived (not-in-memory) session — replay-only, no chatting.
  const [readOnly, setReadOnly] = useState(false);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await listSessions());
    } catch (e) {
      console.error(e);
    }
  }, []);

  // ----- session bootstrap (new, live session) -----
  const startSession = useCallback(async () => {
    setEvents([]);
    setSnapshot(null);
    setPendingUser(null);
    setPendingBot(null);
    setReadOnly(false);
    try {
      const sid = await createSession();
      setSessionId(sid);
      const s = await getState(sid);
      setSnapshot(s);
    } catch (e) {
      console.error(e);
    }
    void refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    void startSession();
  }, [startSession]);

  // ----- load a previous session: replay its stored events EXACTLY as they ran -----
  const loadSession = useCallback(
    async (sid: string, live: boolean) => {
      if (busy) return;
      setBusy(true);
      setPendingUser(null);
      setPendingBot(null);
      try {
        const stored = await fetchEvents(sid);
        // Replay through the SAME projection the live stream uses, so the rebuilt
        // view is identical — every main-agent AND sub-agent row reappears.
        const logs: LogEntry[] = [];
        let snap: AgentSnapshot | null = null;
        for (const row of stored) {
          const ev = row.data;
          for (const entry of logEntriesFor(ev)) {
            entry.ts = formatStoredTs(row.ts) || entry.ts; // show the ORIGINAL time
            logs.push(entry);
          }
          if (ev.type === 'state') snap = ev.snapshot;
        }
        setEvents(logs);
        setSnapshot(snap);
        setSessionId(sid);
        setReadOnly(!live); // archived → read-only; still-live → can keep chatting
      } catch (e) {
        console.error(e);
      } finally {
        setBusy(false);
      }
    },
    [busy],
  );

  // ----- send a turn -----
  const send = useCallback(
    async (msg: string) => {
      if (!sessionId || busy || readOnly) return;
      setBusy(true);
      setPendingUser(msg);
      setPendingBot(null);

      try {
        await streamTurn(sessionId, msg, (ev) => {
          for (const entry of logEntriesFor(ev)) {
            setEvents((prev) => [...prev, entry]);
          }
          if (ev.type === 'state') setSnapshot(ev.snapshot);
          // Stream writer tokens into the pending bubble as they arrive (v4_1).
          if (ev.type === 'token') setPendingBot((p) => (p ?? '') + ev.content);
          if (ev.type === 'writer') setPendingBot(ev.draft);
          if (ev.type === 'bot') setPendingBot(null);
        });
      } catch (e) {
        console.error(e);
        setEvents((prev) => [
          ...prev,
          {
            id: `err-${Date.now()}`,
            ts: new Date().toTimeString().slice(0, 8),
            kind: 'ERROR',
            body: String(e),
          },
        ]);
      } finally {
        setBusy(false);
        setPendingUser(null);
        setPendingBot(null);
        // Refresh authoritative snapshot in case the stream dropped events.
        if (sessionId) {
          try {
            const s = await getState(sessionId);
            setSnapshot(s);
          } catch {}
        }
        void refreshSessions(); // the turn is now persisted — update the picker
      }
    },
    [sessionId, busy, readOnly, refreshSessions],
  );

  return (
    <div className="flex h-screen flex-col">
      <Header
        sessionId={sessionId}
        sessions={sessions}
        readOnly={readOnly}
        onNewSession={startSession}
        onLoadSession={loadSession}
        onOpenSessions={refreshSessions}
        busy={busy}
      />

      <main className="grid flex-1 min-h-0 grid-cols-[minmax(360px,440px)_1fr]">
        {/* Left: cart panel */}
        <aside className="overflow-y-auto border-r border-slate-800/80 bg-slate-950/40">
          <CartPanel snapshot={snapshot} />
        </aside>

        {/* Right: tabs + input */}
        <section className="flex min-h-0 flex-col">
          <div className="flex items-center gap-1 border-b border-slate-800/80 bg-slate-950/40 px-2 py-1.5">
            <TabBtn active={tab === 'events'} onClick={() => setTab('events')}>
              Events
            </TabBtn>
            <TabBtn active={tab === 'chat'} onClick={() => setTab('chat')}>
              Chat
            </TabBtn>
            <div className="ml-auto pr-2 text-[10px] uppercase tracking-wider text-slate-500">
              {events.length} event{events.length === 1 ? '' : 's'}
            </div>
          </div>

          <div className="min-h-0 flex-1">
            {tab === 'events' ? (
              <EventStream entries={events} />
            ) : (
              <ChatPanel snapshot={snapshot} pendingUser={pendingUser} pendingBot={pendingBot} />
            )}
          </div>

          {readOnly ? (
            <div className="flex items-center justify-between gap-2 border-t border-amber-700/40 bg-amber-950/30 px-4 py-2.5 text-xs text-amber-300/90">
              <span>
                📖 Viewing a saved session (read-only). Start a new session to chat.
              </span>
              <button
                onClick={startSession}
                className="shrink-0 rounded-md border border-amber-700/50 bg-amber-900/30 px-3 py-1 font-medium text-amber-200 hover:bg-amber-900/50"
              >
                New session
              </button>
            </div>
          ) : (
            <ChatInput onSend={send} disabled={!sessionId || busy} />
          )}
        </section>
      </main>
    </div>
  );
}

function TabBtn({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
        active
          ? 'bg-slate-800 text-slate-100'
          : 'text-slate-500 hover:bg-slate-900 hover:text-slate-300'
      }`}
    >
      {children}
    </button>
  );
}
