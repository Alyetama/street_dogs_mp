import { useRef, useEffect } from 'react'
import { Link } from 'react-router'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Navigation from '../sections/Navigation'
import FooterSection from '../sections/FooterSection'
import { Terminal, FileOutput, ArrowRight } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

const cliOptions = [
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

const usageExamples = [
  {
    title: 'Run all regions in a CSV',
    code: `python batch_chunks_mp_api_v3.py regions.csv`,
  },
  {
    title: 'Run one specific row by index',
    code: `python batch_chunks_mp_api_v3.py regions.csv --row-index 0`,
  },
  {
    title: 'Fetch metadata without downloading images',
    code: `python batch_chunks_mp_api_v3.py regions.csv --no-download-images`,
  },
  {
    title: 'Download images later from existing Parquets',
    code: `python batch_chunks_mp_api_v3.py regions.csv --download-only`,
  },
  {
    title: 'Send images to a separate disk or mount',
    code: `python batch_chunks_mp_api_v3.py regions.csv --image-dir /path/to/storage`,
  },
  {
    title: 'Use an SSD as a write buffer for a slow HDD',
    code: `python batch_chunks_mp_api_v3.py regions.csv \\
  --image-dir /mnt/hdd/images \\
  --temp-dir /tmp/ssd_spool`,
  },
  {
    title: 'Backfill specific sub-grid indices',
    code: `python batch_chunks_mp_api_v3.py regions.csv --sub-indices 4,12,15`,
  },
  {
    title: 'SLURM array job example',
    code: `python batch_chunks_mp_api_v3.py regions.csv \\
  --slurm \\
  --search-max-workers "$((SLURM_CPUS_PER_TASK * 4))" \\
  --entity-max-workers "$((SLURM_CPUS_PER_TASK * 16))" \\
  --parent-dir grid_runs \\
  --no-download-images`,
  },
]

const outputFiles = [
  { file: 'topology_checkpoint_<sub_id>.json.zst', description: 'Compressed image-to-sequence topology checkpoint for a sub-grid.' },
  { file: 'metadata_checkpoint_<sub_id>.jsonl.zst', description: 'Compressed JSONL metadata checkpoint for all collected image IDs.' },
  { file: 'animal_detections_checkpoint_<sub_id>.jsonl.zst', description: 'Compressed JSONL animal-detection checkpoint.' },
  { file: 'ground_animals_<sub_id>.json.zst', description: 'Compressed detection features for sub-grids where animals were found.' },
  { file: 'all_data_<region>_<part>.parquet', description: 'Chunked Parquet metadata for all images in the region.' },
  { file: 'ground_animals_<region>_<part>.parquet', description: 'Chunked Parquet metadata for images with ground-animal detections.' },
  { file: 'ground_animal_images/', description: 'Downloaded .jpg files when image downloading is enabled.' },
  { file: 'validated_images_<region>.txt', description: 'Ledger of images that passed local validity checks.' },
  { file: 'failed_downloads_<region>.txt', description: 'Image IDs that failed download attempts.' },
  { file: 'covered_countries.txt', description: 'Countries intersecting the region bounding box (written by generate_countries.py).' },
  { file: '.completed_<sub_id>', description: 'Resume marker indicating a sub-grid completed.' },
  { file: '.empty_<sub_id>', description: 'Resume marker indicating a sub-grid had no usable topology results.' },
]

export default function CLIReference() {
  const mainRef = useRef<HTMLElement>(null)
  const tableRef = useRef<HTMLDivElement>(null)
  const examplesRef = useRef<HTMLDivElement>(null)
  const outputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const main = mainRef.current
    if (!main) return

    const sections = [tableRef.current, examplesRef.current, outputRef.current].filter(Boolean)
    const ctx = gsap.context(() => {
      gsap.from(sections, {
        y: 25,
        opacity: 0,
        stagger: 0.12,
        duration: 0.5,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: main,
          start: 'top 80%',
        },
      })
    }, main)

    return () => ctx.revert()
  }, [])

  return (
    <div className="min-h-screen bg-[#1a1a1a]">
      <Navigation />

      <main ref={mainRef} className="pt-24 pb-20">
        <section className="mx-auto max-w-[1000px] px-6 pb-16">
          <div className="mb-6 inline-flex items-center rounded-[20px] border border-[rgba(232,166,69,0.3)] px-3 py-1">
            <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-[#e8a645]">
              CLI Reference
            </span>
          </div>
          <h1 className="text-[clamp(2rem,4vw,3.5rem)] font-normal leading-[1.15] tracking-tight text-[#f8f9fa]">
            <span className="text-[#e8a645]">batch_chunks_mp_api_v3.py</span>
          </h1>
          <p className="mt-5 max-w-[720px] text-[15px] leading-[1.65] text-[#adb5bd]">
            The supported entry point for the pipeline. It processes a CSV of geographic regions through
            a resumable six-phase workflow: tile generation, sequence query, image expansion, metadata
            fetch, detection filtering, and output/download. Re-run the same command to resume from
            compressed checkpoints and completion markers.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              to="/helper-scripts"
              className="group inline-flex items-center gap-2 rounded-md border border-[#495057] bg-transparent px-5 py-2.5 text-[14px] font-medium text-[#f8f9fa] transition-colors duration-200 hover:border-[#adb5bd]"
            >
              Helper Scripts
              <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-1" />
            </Link>
            <a
              href="https://github.com/Alyetama/street_dogs_mp"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md bg-[#e8a645] px-5 py-2.5 text-[14px] font-medium text-[#1a1a1a] transition-colors duration-200 hover:bg-[#f0b85c]"
            >
              View on GitHub
            </a>
          </div>
        </section>

        {/* CLI options table */}
        <section ref={tableRef} className="mx-auto max-w-[1000px] px-6 py-12">
          <div className="mb-8 flex items-center gap-3">
            <Terminal size={22} className="text-[#e8a645]" />
            <h2 className="text-[clamp(1.4rem,2.5vw,1.8rem)] font-normal text-[#f8f9fa]">
              CLI Options
            </h2>
          </div>

          <div className="overflow-x-auto rounded-lg border border-[#495057]">
            <table className="w-full text-left">
              <thead>
                <tr className="bg-[#2c3034]">
                  <th className="px-4 py-3 text-[12px] font-medium uppercase tracking-wider text-[#e8a645]">
                    Option
                  </th>
                  <th className="px-4 py-3 text-[12px] font-medium uppercase tracking-wider text-[#e8a645]">
                    Default
                  </th>
                  <th className="px-4 py-3 text-[12px] font-medium uppercase tracking-wider text-[#e8a645]">
                    Description
                  </th>
                </tr>
              </thead>
              <tbody>
                {cliOptions.map((opt, i) => (
                  <tr
                    key={opt.option}
                    className={`${
                      i % 2 === 0 ? 'bg-[#212529]' : 'bg-[#1e1e1e]'
                    } border-t border-[rgba(73,80,87,0.3)]`}
                  >
                    <td className="whitespace-nowrap px-4 py-3 font-mono text-[13px] text-[#e8a645]">
                      {opt.option}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 font-mono text-[13px] text-[#adb5bd]">
                      {opt.default}
                    </td>
                    <td className="px-4 py-3 text-[14px] text-[#adb5bd]">
                      {opt.description}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-4 text-[13px] text-[#6c757d]">
            For the complete CLI reference, run:{' '}
            <code className="rounded bg-[#2c3034] px-1.5 py-0.5 font-mono text-[12px] text-[#e8a645]">
              python batch_chunks_mp_api_v3.py --help
            </code>
          </p>
        </section>

        {/* Usage examples */}
        <section ref={examplesRef} className="mx-auto max-w-[1000px] px-6 py-12">
          <h2 className="mb-8 text-[clamp(1.4rem,2.5vw,1.8rem)] font-normal text-[#f8f9fa]">
            Usage Examples
          </h2>

          <div className="space-y-6">
            {usageExamples.map((ex) => (
              <div key={ex.title}>
                <h3 className="mb-2 text-[14px] font-medium text-[#f8f9fa]">{ex.title}</h3>
                <div className="rounded-lg bg-[#1e1e1e] border border-[rgba(73,80,87,0.5)] p-4">
                  <pre className="overflow-x-auto font-mono text-[13px] leading-[1.7] text-[#d4d4d4]">
                    <code>{ex.code}</code>
                  </pre>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Output files */}
        <section ref={outputRef} className="mx-auto max-w-[1000px] px-6 py-12">
          <div className="mb-8 flex items-center gap-3">
            <FileOutput size={22} className="text-[#e8a645]" />
            <h2 className="text-[clamp(1.4rem,2.5vw,1.8rem)] font-normal text-[#f8f9fa]">
              Output Files
            </h2>
          </div>

          <p className="mb-6 text-[14px] leading-[1.65] text-[#adb5bd]">
            By default, outputs are written under{' '}
            <code className="rounded bg-[#2c3034] px-1.5 py-0.5 font-mono text-[12px] text-[#e8a645]">
              grid_runs/&lt;safe_region_id&gt;/
            </code>
            . If{' '}
            <code className="rounded bg-[#2c3034] px-1.5 py-0.5 font-mono text-[12px] text-[#e8a645]">
              --image-dir
            </code>{' '}
            is provided, downloaded images are written to{' '}
            <code className="rounded bg-[#2c3034] px-1.5 py-0.5 font-mono text-[12px] text-[#e8a645]">
              &lt;image-dir&gt;/&lt;safe_region_id&gt;/ground_animal_images/
            </code>{' '}
            while checkpoints and Parquet outputs stay under{' '}
            <code className="rounded bg-[#2c3034] px-1.5 py-0.5 font-mono text-[12px] text-[#e8a645]">
              --parent-dir
            </code>
            .
          </p>

          <div className="overflow-x-auto rounded-lg border border-[#495057]">
            <table className="w-full text-left">
              <thead>
                <tr className="bg-[#2c3034]">
                  <th className="px-4 py-3 text-[12px] font-medium uppercase tracking-wider text-[#e8a645]">
                    File / Directory
                  </th>
                  <th className="px-4 py-3 text-[12px] font-medium uppercase tracking-wider text-[#e8a645]">
                    Description
                  </th>
                </tr>
              </thead>
              <tbody>
                {outputFiles.map((out, i) => (
                  <tr
                    key={out.file}
                    className={`${
                      i % 2 === 0 ? 'bg-[#212529]' : 'bg-[#1e1e1e]'
                    } border-t border-[rgba(73,80,87,0.3)]`}
                  >
                    <td className="whitespace-nowrap px-4 py-3 font-mono text-[13px] text-[#e8a645]">
                      {out.file}
                    </td>
                    <td className="px-4 py-3 text-[14px] text-[#adb5bd]">
                      {out.description}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <FooterSection />
    </div>
  )
}
