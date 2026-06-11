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

// ---- Writer "rich reply" blocks (see agent_v4/output_schemas.py) ----
export interface ProductCard {
  id: string;
  name: string;
  price: string;
  tags: string[];
}

export interface OrderCard {
  id: string;
  status: string;
  items: string[];
  tracking_url: string | null;
}

export interface OrderLine {
  id: string;
  name: string;
  qty: number;
  line_total: string;
}

export type Block =
  | { kind: 'product_reco'; products: ProductCard[]; added_ids: string[]; serviceability: string | null }
  | { kind: 'order_status'; order: OrderCard | null; raw: string | null }
  | {
      kind: 'checkout';
      items: OrderLine[];
      subtotal: string | null;
      grand_total: string | null;
      ready_to_confirm: boolean;
      confirmed: boolean;
      receipt_id: string | null;
      asks: string[];
    };

export interface ChatMessage {
  role: string;
  content: string;
  blocks?: Block[];
}

export interface AgentSnapshot {
  user_id: string;
  session_id: string;
  active_sop: SOPName | null;
  skills_loaded: string[];
  cart: Cart;
  messages: ChatMessage[];
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
  | { type: 'writer'; draft: string; blocks?: Block[] }
  | { type: 'gate'; rejected: boolean; errors: string[] }
  | { type: 'validator'; errors: string[] }
  // agent_v4_1 streaming additions (backward-safe; older servers never emit these):
  | { type: 'token'; content: string }
  | { type: 'guardrail'; stage: string; rule: string; action: string }
  | { type: 'bot'; content: string; blocks?: Block[] }
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
