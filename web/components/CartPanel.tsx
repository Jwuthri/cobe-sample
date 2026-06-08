'use client';

import type { AgentSnapshot, CheckoutStep } from '@/lib/types';

const STEPS: { key: CheckoutStep; label: string }[] = [
  { key: 'collecting_products', label: 'Products' },
  { key: 'collecting_identity', label: 'Identity' },
  { key: 'collecting_address', label: 'Address' },
  { key: 'awaiting_serviceability', label: 'Serviceable?' },
  { key: 'collecting_delivery', label: 'Delivery' },
  { key: 'collecting_payment', label: 'Payment' },
  { key: 'ready_to_confirm', label: 'Confirm' },
  { key: 'confirmed', label: 'Done' },
];

function StepIndicator({ current }: { current: CheckoutStep }) {
  const idx = STEPS.findIndex((s) => s.key === current);
  return (
    <div className="flex items-center gap-1 overflow-x-auto py-1">
      {STEPS.map((s, i) => {
        const done = i < idx;
        const active = i === idx;
        return (
          <div key={s.key} className="flex items-center gap-1 shrink-0">
            <div
              className={`flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-medium ${
                active
                  ? 'bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-400/40'
                  : done
                    ? 'bg-emerald-500/15 text-emerald-300'
                    : 'bg-slate-700/40 text-slate-400'
              }`}
            >
              <span className="inline-block size-1.5 rounded-full bg-current" />
              {s.label}
            </div>
            {i < STEPS.length - 1 && <div className="h-px w-2 bg-slate-700/60" />}
          </div>
        );
      })}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 py-1">
      <div className="w-24 shrink-0 text-[11px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className="min-w-0 flex-1 text-sm">{children}</div>
    </div>
  );
}

function Dim({ children }: { children: React.ReactNode }) {
  return <span className="text-slate-500">{children}</span>;
}

export function CartPanel({ snapshot }: { snapshot: AgentSnapshot | null }) {
  if (!snapshot) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-500">
        Start a session to see live state.
      </div>
    );
  }
  const c = snapshot.cart;

  return (
    <div className="flex flex-col gap-4 p-4 text-slate-200">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-500">
            session
          </div>
          <div className="mono text-xs text-slate-300">{snapshot.session_id}</div>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-wider text-slate-500">SOP</div>
          <div className="text-sm font-medium">
            {snapshot.active_sop ?? <Dim>—</Dim>}
          </div>
        </div>
      </div>

      <StepIndicator current={c.step} />

      <div className="rounded-md border border-slate-800/80 bg-slate-900/50 p-3">
        <Row label="customer">
          {c.customer.first_name || c.customer.last_name ? (
            <span>
              {c.customer.first_name ?? ''} {c.customer.last_name ?? ''}
              {c.customer.email && <Dim> · {c.customer.email}</Dim>}
            </span>
          ) : (
            <Dim>—</Dim>
          )}
        </Row>
        <Row label="address">
          {c.address.street ? (
            <span>
              {c.address.street}, {c.address.city ?? '?'}
              {c.address.state ? ` ${c.address.state}` : ''} {c.address.zip_code}{' '}
              <Dim>{c.address.country}</Dim>
            </span>
          ) : (
            <Dim>—</Dim>
          )}
        </Row>
        <Row label="serviceable">
          {c.serviceable === true ? (
            <span className="text-emerald-300">
              yes{' '}
              <Dim>
                (
                {c.serviceable_options.map((opt, i) => (
                  <span key={opt}>
                    {i > 0 && ', '}
                    <span
                      className={
                        opt === c.delivery_option
                          ? 'text-cyan-300 font-medium'
                          : undefined
                      }
                    >
                      {opt}
                    </span>
                  </span>
                ))}
                )
              </Dim>
            </span>
          ) : c.serviceable === false ? (
            <span className="text-rose-300">no</span>
          ) : (
            <Dim>—</Dim>
          )}
        </Row>
        <Row label="delivery">
          {c.delivery_option ? (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan-500/15 px-2 py-0.5 text-[12px] font-medium text-cyan-300 ring-1 ring-cyan-400/30">
              <span className="inline-block size-1.5 rounded-full bg-current" />
              {c.delivery_option}
              {c.shipping && (
                <Dim>
                  {' '}
                  · ${c.shipping.cost} · {c.shipping.eta_hours}h
                </Dim>
              )}
            </span>
          ) : (
            <Dim>—</Dim>
          )}
        </Row>
        <Row label="skills">
          {snapshot.skills_loaded.length ? (
            <div className="flex flex-wrap gap-1">
              {snapshot.skills_loaded.map((s) => (
                <span
                  key={s}
                  className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[11px] text-amber-300"
                >
                  {s}
                </span>
              ))}
            </div>
          ) : (
            <Dim>—</Dim>
          )}
        </Row>
      </div>

      <div className="rounded-md border border-slate-800/80 bg-slate-900/50">
        <div className="flex items-center justify-between border-b border-slate-800/80 px-3 py-2">
          <div className="text-xs uppercase tracking-wider text-slate-500">items</div>
          <div className="text-xs text-slate-400">{c.items.length} line(s)</div>
        </div>
        <div className="p-3">
          {c.items.length === 0 ? (
            <div className="text-sm text-slate-500">(empty)</div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {c.items.map((i) => (
                <div key={i.id} className="flex items-baseline justify-between gap-2 text-sm">
                  <div className="flex items-baseline gap-2 min-w-0">
                    <span className="mono text-xs text-slate-400">{i.id}</span>
                    <span className="truncate">{i.name}</span>
                    <span className="text-xs text-slate-500">×{i.qty}</span>
                  </div>
                  <span className="mono text-xs text-slate-300">${i.line_total}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="border-t border-slate-800/80 px-3 py-2 text-sm">
          <div className="flex justify-between">
            <Dim>subtotal</Dim>
            <span className="mono">${c.subtotal}</span>
          </div>
          <div className="flex justify-between">
            <Dim>shipping</Dim>
            <span className="mono">
              {c.shipping ? (
                <>
                  ${c.shipping.cost} <Dim>· {c.shipping.eta_hours}h</Dim>
                </>
              ) : (
                <span className="text-amber-400/60">stale</span>
              )}
            </span>
          </div>
          <div className="flex justify-between">
            <Dim>tax</Dim>
            <span className="mono">
              {c.tax ? (
                <>${c.tax.amount}</>
              ) : (
                <span className="text-amber-400/60">stale</span>
              )}
            </span>
          </div>
          {c.promo && (
            <div className="flex justify-between text-emerald-300">
              <Dim>
                promo <span className="mono text-emerald-300">{c.promo.code}</span>
              </Dim>
              <span className="mono">−${c.promo.discount}</span>
            </div>
          )}
          <div className="mt-1.5 flex justify-between border-t border-slate-800/80 pt-1.5 text-base">
            <span className="font-medium">grand total</span>
            <span className="mono font-semibold">
              {c.grand_total ? `$${c.grand_total}` : <Dim>—</Dim>}
            </span>
          </div>
        </div>
      </div>

      <div className="rounded-md border border-slate-800/80 bg-slate-900/50 p-3">
        <Row label="payment">
          {c.payment_method ? (
            <span>
              {c.payment_method}{' '}
              <Dim>
                {c.payment_method === 'card'
                  ? c.card_token_set
                    ? '(tok set)'
                    : '(no tok)'
                  : ''}
              </Dim>
            </span>
          ) : (
            <Dim>—</Dim>
          )}
        </Row>
        <Row label="blockers">
          {c.blockers.length ? (
            <ul className="flex flex-col gap-0.5">
              {c.blockers.map((b) => (
                <li key={b.code} className="flex items-baseline gap-2 text-sm">
                  <span className="mono text-[11px] text-rose-300">{b.code}</span>
                  <span className="text-slate-400">{b.message}</span>
                </li>
              ))}
            </ul>
          ) : (
            <span className="text-emerald-300">none ✓</span>
          )}
        </Row>
        {c.confirmed && c.receipt_id && (
          <Row label="receipt">
            <span className="mono text-emerald-300">{c.receipt_id}</span>
          </Row>
        )}
      </div>
    </div>
  );
}
