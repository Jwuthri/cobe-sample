'use client';

// Renderers for the writer's typed "rich reply" blocks (see
// agent_v4/output_schemas.py). The chat message text renders in the bubble;
// these structured cards render beneath it. Unknown kinds degrade to nothing.

import type { Block, OrderCard, ProductCard } from '@/lib/types';

export function StructuredBlocks({ blocks }: { blocks?: Block[] | null }) {
  if (!blocks || blocks.length === 0) return null;
  return (
    <div className="flex w-full flex-col gap-2">
      {blocks.map((b, i) => (
        <BlockView key={i} block={b} />
      ))}
    </div>
  );
}

function BlockView({ block }: { block: Block }) {
  switch (block.kind) {
    case 'product_reco':
      return <ProductReco block={block} />;
    case 'order_status':
      return <OrderStatus block={block} />;
    case 'checkout':
      return <Checkout block={block} />;
    default:
      return null; // graceful degrade for unknown/older block kinds
  }
}

function Card({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-slate-800/80 bg-slate-900/50 p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
      {children}
    </div>
  );
}

function ProductReco({ block }: { block: Extract<Block, { kind: 'product_reco' }> }) {
  return (
    <Card label="products">
      <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        {block.products.map((p) => (
          <ProductCardView key={p.id} p={p} />
        ))}
      </div>
      {block.serviceability && (
        <div className="mt-2 text-xs text-slate-400">{block.serviceability}</div>
      )}
    </Card>
  );
}

function ProductCardView({ p }: { p: ProductCard }) {
  return (
    <div className="flex items-baseline justify-between gap-2 rounded border border-slate-800/60 bg-slate-950/40 px-2 py-1.5">
      <div className="min-w-0">
        <div className="flex items-baseline gap-1.5">
          <span className="mono text-[11px] text-slate-400">{p.id}</span>
          <span className="truncate text-sm text-slate-100">{p.name}</span>
        </div>
        {p.tags.length > 0 && (
          <div className="mt-0.5 flex flex-wrap gap-1">
            {p.tags.map((t) => (
              <span key={t} className="rounded bg-slate-700/40 px-1 py-0.5 text-[10px] text-slate-400">
                {t}
              </span>
            ))}
          </div>
        )}
      </div>
      <span className="mono shrink-0 text-sm text-slate-200">${p.price}</span>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === 'delivered'
      ? 'bg-emerald-500/15 text-emerald-300'
      : status === 'shipped'
        ? 'bg-cyan-500/15 text-cyan-300'
        : 'bg-slate-700/40 text-slate-300';
  return <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${tone}`}>{status}</span>;
}

function OrderStatus({ block }: { block: Extract<Block, { kind: 'order_status' }> }) {
  const o: OrderCard | null = block.order;
  if (!o) {
    return <Card label="order">{block.raw && <div className="text-xs text-slate-400">{block.raw}</div>}</Card>;
  }
  return (
    <Card label="order">
      <div className="flex items-center justify-between">
        <span className="mono text-sm text-slate-200">{o.id}</span>
        <StatusBadge status={o.status} />
      </div>
      <div className="mt-1 text-xs text-slate-400">items: {o.items.join(', ')}</div>
      {o.tracking_url && (
        <a
          href={o.tracking_url}
          target="_blank"
          rel="noreferrer"
          className="mt-1 inline-block text-xs text-sky-400 hover:underline"
        >
          track →
        </a>
      )}
    </Card>
  );
}

function Checkout({ block }: { block: Extract<Block, { kind: 'checkout' }> }) {
  return (
    <Card label="order summary">
      <div className="flex flex-col gap-1">
        {block.items.map((li) => (
          <div key={li.id} className="flex items-baseline justify-between gap-2 text-sm">
            <div className="flex min-w-0 items-baseline gap-2">
              <span className="mono text-xs text-slate-400">{li.id}</span>
              <span className="truncate text-slate-100">{li.name}</span>
              <span className="text-xs text-slate-500">×{li.qty}</span>
            </div>
            <span className="mono text-xs text-slate-300">${li.line_total}</span>
          </div>
        ))}
      </div>
      <div className="mt-1.5 flex justify-between border-t border-slate-800/80 pt-1.5">
        <span className="font-medium">{block.grand_total ? 'grand total' : 'subtotal'}</span>
        <span className="mono font-semibold">${block.grand_total ?? block.subtotal ?? '—'}</span>
      </div>
      {block.confirmed && block.receipt_id ? (
        <div className="mt-1 text-xs text-emerald-300">
          confirmed · <span className="mono">{block.receipt_id}</span>
        </div>
      ) : block.ready_to_confirm ? (
        <div className="mt-1 text-xs text-cyan-300">Ready to place — reply &quot;yes&quot; to confirm.</div>
      ) : block.asks.length > 0 ? (
        <div className="mt-1 text-xs text-amber-300">needs: {block.asks.join(', ')}</div>
      ) : null}
    </Card>
  );
}
