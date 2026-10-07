import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    globalSetup: ["test/fixtures.ts"],
    testTimeout: 30_000,
  },
});
