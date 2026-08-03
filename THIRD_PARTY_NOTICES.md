# Third-party notices

This project is MIT licensed (see [LICENSE](LICENSE)). It redistributes the
following third-party component, which keeps its own license.

## Apache ECharts

- **File:** `tools/dashboard/echarts.min.js`
- **Version:** 5.6.0
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

## Not redistributed

`data/dashboard/world.json`, the country geometry the dashboard's map registers
with ECharts, is **not** in this repository. Supply your own; whatever you use
carries its own license and attribution requirements, which are yours to meet.
Without it the map does not render and the rest of the dashboard is unaffected.
