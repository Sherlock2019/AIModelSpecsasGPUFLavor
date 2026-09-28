import { defineConfig } from '@playwright/test'

// Optional end-to-end tests. Start the API and `npm run dev` first, then `npm run e2e`.
export default defineConfig({
  testDir: 'e2e',
  timeout: 30_000,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:5173',
    viewport: { width: 1440, height: 1000 },
  },
})
