# Documentation site

The marketing / documentation site for **street_dogs_mp**, built with React 19,
Vite 7, TypeScript, and Tailwind CSS. It documents the three-stage pipeline
(Extract → Audit → Backfill) and the helper-script catalog.

## Develop

```bash
cd docs
npm install
npm run dev      # local dev server with HMR
```

## Build & preview

```bash
npm run build    # type-checks (tsc -b) then bundles to dist/
npm run preview  # serve the production build locally
```

## Structure

| Path | Purpose |
| --- | --- |
| `src/pages/` | Routed pages: `Home`, `CLIReference` (Extract), `CoverageAudit`, `Backfill`, `HelperScripts`. |
| `src/sections/` | Home sections + the shared `ReferencePage` used by the three stage pages. |
| `src/App.tsx` | Routes. |
| `src/components/ui/` | shadcn/ui primitives. |

The three stage reference pages (`CLIReference`, `CoverageAudit`, `Backfill`)
all render through `src/sections/ReferencePage.tsx` — edit that component to
change their shared layout, and the per-page data files for their content.

## Deploy

The app uses `HashRouter` under the base path `/street_dogs_mp/` (set in
`vite.config.ts`), so routes live behind a `#` (e.g.
`/street_dogs_mp/#/coverage-audit`) and **no SPA rewrites are needed** — any
static host serving `dist/` at that base path works (e.g. GitHub Pages). If you
deploy at a different base, update `base` in `vite.config.ts`.
