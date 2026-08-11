import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { inspectAttr } from 'kimi-plugin-inspect-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  base: '/street_dogs_mp/',
  // inspectAttr() is a dev affordance: it stamps code-path="src/<file>:<line>:<col>"
  // on every JSX element so a click can be traced back to its source. Its README
  // says it only applies to `vite serve`, but the pinned 1.0.3 ships a plugin
  // object with no `apply` key, so registering it unconditionally ran the Babel
  // transform over the PRODUCTION build too -- 270 source locations baked into
  // the published bundle and 181-520 code-path attributes in the DOM of every
  // deployed page. Gated here rather than by upgrading, so the guarantee does
  // not depend on the plugin's own defaults.
  plugins: [...(command === 'serve' ? [inspectAttr()] : []), react()],
  server: {
    port: 3000,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
