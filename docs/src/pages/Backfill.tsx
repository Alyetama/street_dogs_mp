import ReferencePage from '../sections/ReferencePage'

const options = [
  { option: '--inscope', default: 'coverage_missing_inscope', description: 'Datefilter output directory from the audit stage (or a single parquet).' },
  { option: '--region', default: 'all', description: 'Parent region to backfill; accepts the sanitized name.' },
  { option: '--out-dir', default: 'grid_runs', description: 'Where backfill parquets are written.' },
  { option: '--image-dir', default: '= out-dir', description: 'Separate drive/root for the downloaded jpgs.' },
  { option: '--no-download', default: 'False', description: 'Write parquets only; download the images later.' },
  { option: '--download-only', default: 'False', description: 'Skip metadata; only download jpgs from existing ground_animals_* backfill parquets.' },
  { option: '--watch', default: 'False', description: 'In --download-only, keep re-scanning for new parquets instead of exiting when caught up.' },
  { option: '--processes', default: '1', description: 'Fan out across N OS processes (disjoint token/row slices) to beat the single-process JSON-parse ceiling.' },
  { option: '--entity-workers', default: '520', description: 'Threads for metadata/detection calls. The entity API is 60,000/min per token.' },
  { option: '--download-workers', default: '10', description: 'Threads for image downloads.' },
  { option: '--batch', default: '50000', description: 'Image ids fetched per round (larger amortizes the per-round straggler barrier).' },
  { option: '--proxies', default: 'unset', description: 'Rotating proxies for metadata fetches.' },
]

const examples = [
  {
    title: 'Backfill a region — parquets and images on separate drives',
    code: `python backfill_missing.py \\
  --inscope coverage_missing_inscope --region Europe \\
  --out-dir /path/to/grid_runs --image-dir /path/to/images \\
  --processes 3 --entity-workers 520 --download-workers 10`,
  },
  {
    title: 'Metadata only now, download later',
    code: `python backfill_missing.py \\
  --inscope coverage_missing_inscope --region Europe \\
  --out-dir /path/to/grid_runs --no-download --processes 3`,
  },
  {
    title: 'Download images for an already-backfilled region',
    code: `python backfill_missing.py --download-only \\
  --out-dir /path/to/grid_runs --image-dir /path/to/images \\
  --download-workers 10`,
  },
  {
    title: 'Continuously drain images while a metadata scan runs',
    code: `python backfill_missing.py --download-only --watch \\
  --out-dir /path/to/grid_runs --image-dir /path/to/images`,
  },
]

const outputs = [
  { file: 'all_data_<cell>_backfill_<NNN>.parquet', description: 'Metadata for every backfilled image in the cell (for later stats).' },
  { file: 'ground_animals_<cell>_backfill_<NNN>.parquet', description: 'Metadata for the ground-animal images in the cell.' },
  { file: '<image-dir>/<cell>/ground_animal_images/<id>.jpg', description: 'Downloaded ground-animal jpgs (skipped under --no-download).' },
  { file: '.backfill_progress[.sIofN].json', description: 'Per-parent row-offset resume sidecar (one per shard).' },
]

const lead =
  'Step 4 of the pipeline — fetches the in-scope missing set produced by the audit. For each image it gets metadata and detections in one Graph API call, writes append-only Parquet chunks matching the main pipeline schema (both an all_data_* and a ground_animals_* file), and downloads the ground-animal jpgs. --out-dir and --image-dir can point at different drives.'

const outputIntro = (
  <>
    Resumable: the in-scope parquet is processed sequentially with a per-parent
    row offset recorded in the progress sidecar, and per-cell part numbers
    continue past existing files. A finished region's offset sits at the end,
    so re-running won't re-download skipped images — use{' '}
    <code className="rounded bg-[#2c3034] px-1.5 py-0.5 font-mono text-[12px] text-[#e8a645]">
      --download-only
    </code>{' '}
    to fetch images for a region whose metadata is already complete.
  </>
)

export default function Backfill() {
  return (
    <ReferencePage
      badge="Stage 3 · Backfill"
      title="backfill_missing.py"
      lead={lead}
      primaryLink={{ label: 'Helper scripts', to: '/helper-scripts' }}
      options={options}
      optionsHelp="python backfill_missing.py --help"
      examples={examples}
      outputIntro={outputIntro}
      outputs={outputs}
    />
  )
}
