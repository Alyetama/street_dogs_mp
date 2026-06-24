import ReferencePage from '../sections/ReferencePage'

const options = [
  { option: '--zoom-level', default: '14', description: 'Mercantile tile zoom used while scanning bounding boxes.' },
  { option: '--outer-max-workers', default: '5', description: 'Number of regions processed in parallel in local mode.' },
  { option: '--search-max-workers', default: '150 / outer', description: 'Threads per region for bbox/sequence search API calls.' },
  { option: '--entity-max-workers', default: '520 / outer', description: 'Threads per region for metadata and detection API calls.' },
  { option: '--download-max-workers', default: '10', description: 'Threads used specifically for image downloads.' },
  { option: '--sub-grid-step', default: '1.0', description: 'Degree step used to split large input regions into smaller sub-grids.' },
  { option: '--sub-indices', default: 'unset', description: 'Comma-separated sub-grid indices to process (e.g., 4 or 4,12,15).' },
  { option: '--parent-dir', default: 'grid_runs', description: 'Root directory for per-region outputs and checkpoints.' },
  { option: '--image-dir', default: 'unset', description: 'Separate directory for image downloads (e.g., a different disk).' },
  { option: '--temp-dir', default: 'unset', description: 'Fast SSD temp directory to buffer downloads before moving to --image-dir.' },
  { option: '--api-chunk-size', default: '5000', description: 'Batch size for threaded API work.' },
  { option: '--parquet-chunk-size', default: '100000', description: 'Rows per all_data_*.parquet output partition.' },
  { option: '--proxy-file', default: 'unset', description: 'Optional proxy list file. Supports ip:port and full proxy URLs.' },
  { option: '--exclude-ledger', default: 'unset', description: 'Text file of image IDs to skip.' },
  { option: '--token', default: 'unset', description: 'Selects MLY_KEY_<n> instead of MLY_KEY.' },
  { option: '--slurm', default: 'False', description: 'Runs one region based on SLURM_ARRAY_TASK_ID or --row-index.' },
  { option: '--download-only', default: 'False', description: 'Skip API fetching and ONLY download images from existing Parquets.' },
  { option: '--no-download-images', default: 'False', description: 'Fetch metadata and detections without downloading images.' },
  { option: '--row-index', default: 'unset', description: 'Run one specific CSV row by zero-based index.' },
]

const examples = [
  { title: 'Run all regions in a CSV', code: `python batch_chunks_mp_api.py regions.csv` },
  { title: 'Run one specific row by index', code: `python batch_chunks_mp_api.py regions.csv --row-index 0` },
  { title: 'Fetch metadata without downloading images', code: `python batch_chunks_mp_api.py regions.csv --no-download-images` },
  { title: 'Download images later from existing Parquets', code: `python batch_chunks_mp_api.py regions.csv --download-only` },
  { title: 'Send images to a separate disk or mount', code: `python batch_chunks_mp_api.py regions.csv --image-dir /path/to/storage` },
  {
    title: 'Use an SSD as a write buffer for a slow HDD',
    code: `python batch_chunks_mp_api.py regions.csv \\
  --image-dir /mnt/hdd/images \\
  --temp-dir /tmp/ssd_spool`,
  },
  { title: 'Backfill specific sub-grid indices', code: `python batch_chunks_mp_api.py regions.csv --sub-indices 4,12,15` },
  {
    title: 'SLURM array job example',
    code: `python batch_chunks_mp_api.py regions.csv \\
  --slurm \\
  --search-max-workers "$((SLURM_CPUS_PER_TASK * 4))" \\
  --entity-max-workers "$((SLURM_CPUS_PER_TASK * 16))" \\
  --parent-dir grid_runs \\
  --no-download-images`,
  },
]

const outputs = [
  { file: 'topology_checkpoint_<sub_id>.json.zst', description: 'Compressed image-to-sequence topology checkpoint for a sub-grid.' },
  { file: 'metadata_checkpoint_<sub_id>.jsonl.zst', description: 'Compressed JSONL metadata checkpoint for all collected image IDs.' },
  { file: 'animal_detections_checkpoint_<sub_id>.jsonl.zst', description: 'Compressed JSONL animal-detection checkpoint.' },
  { file: 'ground_animals_<sub_id>.json.zst', description: 'Compressed detection features for sub-grids where animals were found.' },
  { file: 'all_data_<region>_<part>.parquet', description: 'Chunked Parquet metadata for all images in the region.' },
  { file: 'ground_animals_<region>_<part>.parquet', description: 'Chunked Parquet metadata for images with ground-animal detections.' },
  { file: 'ground_animal_images/', description: 'Downloaded .jpg files when image downloading is enabled.' },
  { file: 'validated_images_<region>.txt', description: 'Ledger of images that passed local validity checks.' },
  { file: 'failed_downloads_<region>.txt', description: 'Image IDs that failed download attempts.' },
  { file: 'covered_countries.txt', description: 'Countries intersecting the region bounding box (written by tools/grid/generate_countries.py).' },
  { file: '.completed_<sub_id>', description: 'Resume marker indicating a sub-grid completed.' },
  { file: '.empty_<sub_id>', description: 'Resume marker indicating a sub-grid had no usable topology results.' },
]

const lead =
  'The supported entry point for the pipeline. It processes a CSV of geographic regions through a resumable six-phase workflow: tile generation, sequence query, image expansion, metadata fetch, detection filtering, and output/download. Re-run the same command to resume from compressed checkpoints and completion markers.'

const code = (s: string) => (
  <code className="rounded bg-[#2c3034] px-1.5 py-0.5 font-mono text-[12px] text-[#e8a645]">
    {s}
  </code>
)

const outputIntro = (
  <>
    By default, outputs are written under {code('grid_runs/<safe_region_id>/')}. If{' '}
    {code('--image-dir')} is provided, downloaded images go to{' '}
    {code('<image-dir>/<safe_region_id>/ground_animal_images/')} while checkpoints
    and Parquet outputs stay under {code('--parent-dir')}.
  </>
)

export default function CLIReference() {
  return (
    <ReferencePage
      badge="Stage 1 · Extract"
      title="batch_chunks_mp_api.py"
      lead={lead}
      primaryLink={{ label: 'Audit stage', to: '/coverage-audit' }}
      options={options}
      optionsHelp="python batch_chunks_mp_api.py --help"
      examples={examples}
      outputIntro={outputIntro}
      outputs={outputs}
    />
  )
}
