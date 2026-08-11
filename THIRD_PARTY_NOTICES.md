# Third-party notices

This project is MIT licensed (see [LICENSE](LICENSE)). It redistributes the
following third-party components, which keep their own licenses.

## Apache ECharts

- **File:** `tools/dashboard/echarts.min.js`
- **Version:** 5.5.1
- **Copyright:** The Apache Software Foundation
- **License:** Apache License 2.0 — <https://www.apache.org/licenses/LICENSE-2.0>
- **Upstream:** <https://github.com/apache/echarts>

Vendored rather than loaded from a CDN because the dashboard is served on a
private interface and must render with no outbound network access. The file is
the unmodified upstream minified bundle and retains its Apache-2.0 header,
which is what satisfies the license's attribution requirement; this file exists
to make that attribution findable without reading a minified bundle.

Apache ECharts is not covered by this project's MIT license, and the MIT
license does not apply to it.

> The bundle carries two version strings. `version:"5.6.0"` belongs to the
> zrender renderer bundled inside it; ECharts' own export is
> `t.version="5.5.1"`, which is the number above. Reading the first one is how
> this notice came to name a release the file is not: verify with
> `grep -o '\.version="5\.[0-9.]*"'`, and the file is byte-identical to
> upstream `echarts@5.5.1/dist/echarts.min.js`
> (sha256 `e84270bd0cd5bdf60fefc26d00c2a391cb2e81f4d26a7a9ee16185a54773a3cf`,
> 1,030,855 bytes).

## Geist and Geist Mono

- **Files:** `docs/public/fonts/Geist-Variable.woff2`,
  `docs/public/fonts/GeistMono-Variable.woff2`
- **Copyright:** © 2023 Vercel, in collaboration with basement.studio
- **License:** SIL Open Font License 1.1 —
  [`docs/public/fonts/OFL.txt`](docs/public/fonts/OFL.txt),
  <https://openfontlicense.org/>
- **Upstream:** <https://github.com/vercel/geist-font>

Vendored for the same reason as ECharts: the docs site loads them from its own
origin (`docs/src/index.css`) rather than a font CDN. They are the unmodified
variable `.woff2` files from the `geist` npm package. The OFL requires that
every redistributed copy travel with the copyright notice and the license, so
`OFL.txt` sits beside them in the same directory and is deployed with them.

Geist is not covered by this project's MIT license, and the MIT license does
not apply to it.

## Not redistributed

`data/dashboard/world.json`, the country geometry the dashboard's map registers
with ECharts, is **not** in this repository. Supply your own; whatever you use
carries its own license and attribution requirements, which are yours to meet.
Without it the map does not render and the rest of the dashboard is unaffected.
