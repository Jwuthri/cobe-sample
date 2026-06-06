import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'agent_v2 — debug console',
  description: 'Multi-agent graph debugger',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
