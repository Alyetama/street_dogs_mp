import ReferencePage from '../sections/ReferencePage'

const subcommands = [
  { name: 'audit', description: 'The normal entry point — runs enumerate → retry → diff → datefilter in one command.' },
  { name: 'enumerate', description: 'Fetch every z14 land tile of each region as a vector tile; write a per-region checkpoint and a tiny meta sidecar. Resumable across days.' },
  { name: 'check', description: 'Read only the meta sidecars (instant); report per-region image counts and failed-tile counts.' },
  { name: 'retry', description: "Re-fetch only each region's failed tiles and merge them in. Safe to run repeatedly." },
  { name: 'diff', description: 'missing = coverage_ids − all_data_ids → per-parent Parquet shards of missing image_ids (carrying captured_at).' },
  { name: 'datefilter', description: 'Keep missing rows captured on/before --cutoff. Local when captured_at is present; falls back to the Graph API only for rows lacking it.' },
]

const options = [
  { option: '--region', default: 'all', description: 'Restrict to one parent region. Accepts the original or sanitized name ("Middle East" or Middle_East).' },
  { option: '--data-dir', default: '.', description: 'Where coverage checkpoints / meta and the budget sidecar live (the data drive, not image dirs).' },
  { option: '--dirs', default: '—', description: 'Base directories holding all_data_*.parquet (used by diff).' },
  { option: '-w / --workers', default: '64', description: 'Concurrent tile fetches.' },
  { option: '--entity-workers', default: '520', description: 'Concurrency for the datefilter API fallback.' },
  { option: '--cutoff', default: '2026-05-31', description: 'Keep images captured on or before this date (audit / datefilter).' },
  { option: '--outer-workers', default: '1', description: 'Process this many regions at once, each with a disjoint slice of tokens and proxies.' },
  { option: '--proxies', default: 'unset', description: 'Rotating proxy list for tile/API requests (the tiles throttle is per-IP).' },
  { option: '--daily-tile-limit', default: '50000', description: 'Per-token daily tile cap, tracked in .tile_request_budget.json.' },
  { option: '--wait', default: 'False', description: 'When the daily budget is exhausted, probe for the reset and auto-resume (timezone-agnostic).' },
  { option: '--wait-interval', default: '1800', description: 'Seconds between budget-reset probes when --wait is set.' },
  { option: '--missing-out', default: 'coverage_missing', description: 'Output directory for diff shards.' },
  { option: '--inscope-out', default: 'coverage_missing_inscope', description: 'Output directory for the datefilter in-scope parquets (the backfill input).' },
]

const examples = [
  {
    title: 'Full audit of one region, reading data spread across drives',
    code: `python coverage_audit.py audit original_global_grid_5deg.csv \\
  --dirs grid_runs /mnt/hdd/grid_runs --region Europe`,
  },
  {
    title: 'Enumerate only (write coverage checkpoints)',
    code: `python coverage_audit.py enumerate original_global_grid_5deg.csv --region Europe -w 64`,
  },
  {
    title: 'Check status instantly from the meta sidecars',
    code: `python coverage_audit.py check --region Europe`,
  },
  {
    title: 'Retry only the failed tiles',
    code: `python coverage_audit.py retry --region Europe`,
  },
  {
    title: 'Diff against extracted data, then keep the in-scope set',
    code: `python coverage_audit.py diff original_global_grid_5deg.csv --dirs grid_runs --region Europe
python coverage_audit.py datefilter --cutoff 2026-05-31`,
  },
  {
    title: 'Multi-day run: wait for the daily budget reset and auto-resume',
    code: `python coverage_audit.py audit original_global_grid_5deg.csv \\
  --dirs grid_runs --region Europe --wait`,
  },
]

const outputs = [
  { file: 'coverage_checkpoint_<region>.json.zst', description: '{ image_id: [sequence_id, captured_at_ms] } for every image enumerated in the region.' },
  { file: 'coverage_meta_<region>.json', description: 'Small sidecar (method, errors, failed_tiles, n_images) so check / retry are instant.' },
  { file: '.tile_request_budget.json', description: 'Per-token daily tile-request usage, reset by date.' },
  { file: 'coverage_missing/<Parent>/<cell>.parquet', description: 'diff output — per-cell shards of missing image_ids (with captured_at).' },
  { file: 'coverage_missing_inscope/<Parent>.parquet', description: 'datefilter output — the in-scope missing set; this is what backfill consumes.' },
]

const lead =
  'Completeness audit: verifies the extraction captured every Mapillary image (up to a cutoff date) for each grid region, and lists what is missing. It enumerates via Mapillary vector tiles — not the /images?bbox search API, which silently caps at ~2000 results per query and 500s on dense city cells — so a single z14 tile returns every image point with no such limit.'

export default function CoverageAudit() {
  return (
    <ReferencePage
      badge="Stage 2 · Audit"
      title="coverage_audit.py"
      lead={lead}
      primaryLink={{ label: 'Backfill stage', to: '/backfill' }}
      subcommands={{
        intro: 'Run the whole audit at once with the audit subcommand, or run any stage on its own. tiles.mapillary.com allows ~50,000 requests/day per token, so a continental enumeration spans multiple days — just re-run the same command (or pass --wait) and completed regions are skipped.',
        items: subcommands,
      }}
      options={options}
      optionsHelp="python coverage_audit.py audit --help"
      examples={examples}
      outputs={outputs}
    />
  )
}
