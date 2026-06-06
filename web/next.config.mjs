/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy /api/* to the FastAPI server so the browser can talk to it
  // without CORS surprises and so the same origin serves both UI and API.
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.AGENT_V2_API_URL ?? 'http://localhost:8001'}/api/:path*`,
      },
    ];
  },
};
export default nextConfig;
