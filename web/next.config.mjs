// Sends any browser request to /api/... on to the FastAPI service, so the website and the
// API feel like one app (and there are no cross-site request headaches).
const API_URL = process.env.API_URL || "http://127.0.0.1:8000";

export default {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};
