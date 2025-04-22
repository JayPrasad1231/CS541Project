import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: [
      "i.pravatar.cc", // for user avatar
      "cdn-icons-png.flaticon.com", // for bot avatar
    ],
  },
};

export default nextConfig;
