// Shared types matching the SSE event shapes emitted by server/main.py.

export type SOPName = 'product_rec' | 'checkout' | 'order_status';

export type CheckoutStep =
  | 'collecting_products'
  | 'collecting_identity'
  | 'collecting_address'
  | 'awaiting_serviceability'
  | 'collecting_delivery'
  | 'collecting_payment'
  | 'ready_to_confirm'
  | 'confirmed';

export interface Blocker {
  code: string;
  message: string;
}

export interface CartItem {
  id: string;
  name: string;
  qty: number;
  unit_price: string;
  line_total: string;
  tags: string[];
}

export interface Cart {
  step: CheckoutStep;
  cart_id: string | null;
  items: CartItem[];
  customer: { first_name: string | null; last_name: string | null; email: string | null };
  address: {
    street: string | null;
    city: string | null;
    state: string | null;
    zip_code: string | null;
    country: string;
  };
  serviceable: boolean | null;
  serviceable_options: string[];
  delivery_option: string | null;
  shipping: { cost: string; eta_hours: number } | null;
  tax: { amount: string; rate: string } | null;
  promo: { code: string; discount: string } | null;
  payment_method: string | null;
  card_token_set: boolean;
  subtotal: string;
  grand_total: string | null;
  blockers: Blocker[];
  ready_to_confirm: boolean;
  confirmed: boolean;
  receipt_id: string | null;
}

export interface AgentSnapshot {
  user_id: string;
  session_id: string;
  active_sop: SOPName | null;
  skills_loaded: string[];
  cart: Cart;
  messages: { role: string; content: string }[];
  iteration: number;
  done: boolean;
}

// ---- SSE events emitted by the server ----
export type ServerEvent =
  | { type: 'user'; content: string }
  | { type: 'state'; snapshot: AgentSnapshot }
  | { type: 'router'; target: string; iteration: number }
  | { type: 'agent'; node: string }
  | { type: 'skill'; name: string }
  | { type: 'tool_start'; name: string; args: Record<string, unknown> }
  | { type: 'tool_end'; name: string; result: string }
  | { type: 'step'; sop: SOPName; summary: string; asks: string[]; next_sop: SOPName | null; details: Record<string, unknown> | null }
  | { type: 'writer'; draft: string }
  | { type: 'gate'; rejected: boolean; errors: string[] }
  | { type: 'validator'; errors: string[] }
  | { type: 'bot'; content: string }
  | { type: 'end' }
  | { type: 'error'; content: string };

// Log entries we surface in the event-stream panel (richer than raw events
// because we add timestamps + presentation metadata).
export interface LogEntry {
  id: string;
  ts: string;
  kind:
    | 'USER'
    | 'ROUTER'
    | 'AGENT'
    | 'SKILL'
    | 'TOOL'
    | 'TOOL_RESULT'
    | 'STEP'
    | 'WRITER'
    | 'GATE'
    | 'VALIDATOR'
    | 'BOT'
    | 'ERROR';
  body: string;
  payload?: Record<string, unknown>;
}
