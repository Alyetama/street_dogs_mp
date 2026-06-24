import { useRef, useEffect } from 'react'
import { Link } from 'react-router'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Navigation from '../sections/Navigation'
import FooterSection from '../sections/FooterSection'
import { Wrench, ArrowRight } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

type Script = {
  name: string
  description: string
  usage: string
  examples?: string[]
  options?: { option: string; default: string; description: string }[]
}

const categories: { title: string; scripts: Script[] }[] = [
  {
    title: 'Grid Preparation',
    scripts: [
      {
        name: 'split_regions.py',
        description:
          'Splits global_grid_5deg.csv into individual per-region CSV files placed under regions/pending/. Each file contains the rows for a single named region and is ready to pass directly to batch_chunks_mp_api.py.',
        usage: 'python split_regions.py',
        examples: ['python split_regions.py'],
      },
      {
        name: 'generate_countries.py',
        description:
          'For every region folder found under one or more --dirs directories, writes a covered_countries.txt file listing the countries whose boundaries intersect that region\'s bounding box. Uses Natural Earth 110 m country data.',
        usage: 'python generate_countries.py --dirs <dir> [<dir> ...]',
        examples: [
          'python generate_countries.py --dirs grid_runs',
          'python generate_countries.py --dirs grid_runs /mnt/hdd/grid_runs',
        ],
      },
    ],
  },
  {
    title: 'Progress & Navigation',
    scripts: [
      {
        name: 'progress_tracker.py',
        description:
          'Displays a Rich-formatted progress table grouped by parent region, showing completion percentage, total data points, ground-animal counts, and image download rate. Accepts multiple base directories so runs spread across several drives can be reported together. Saves a timestamped CSV report to progress_files/.',
        usage: 'python progress_tracker.py <regions.csv> --dirs <dir> [<dir> ...] [options]',
        examples: [
          'python progress_tracker.py regions.csv --dirs grid_runs',
          'python progress_tracker.py regions.csv --dirs grid_runs /mnt/hdd/grid_runs',
        ],
        options: [
          { option: '--dirs', default: 'required', description: 'One or more base directories to scan for grid runs.' },
          { option: '--sub-grid-step', default: '1.0', description: 'Must match the --sub-grid-step used in the main script.' },
          { option: '-w / --workers', default: '2 × CPU', description: 'Concurrent threads for scanning.' },
        ],
      },
      {
        name: 'find_location_folder.py',
        description:
          'Geocodes a city or country name via Nominatim and finds which region folders in your grid runs overlap with it. Useful for quickly locating data for a specific place across multiple drives.',
        usage: 'python find_location_folder.py "<place>" --dirs <dir> [<dir> ...]',
        examples: [
          'python find_location_folder.py "Japan" --dirs grid_runs /mnt/hdd/grid_runs',
          'python find_location_folder.py "Paris" --dirs grid_runs',
        ],
      },
      {
        name: 'scan_regions.py',
        description:
          'Scans one or more base directories for folders matching a region prefix, then recommends the exact --parent-dir and --image-dir flags to pass to the main script. Also reports per-directory breakdowns and flags any regions whose data or images are split across unexpected directories.',
        usage: 'python scan_regions.py <region_prefix> --dirs <dir> [<dir> ...]',
        examples: [
          'python scan_regions.py South_America --dirs grid_runs /mnt/hdd/grid_runs',
        ],
      },
    ],
  },
  {
    title: 'Ledger Management',
    scripts: [
      {
        name: 'generate_ledger.py',
        description:
          'Builds or appends to an exclude ledger (a plain-text file of image IDs) by scanning a directory tree for .jpg files. Pass the resulting file to the main script via --exclude-ledger to skip images that have already been downloaded.',
        usage: 'python generate_ledger.py --image-dir <dir> --output <ledger.txt> [options]',
        examples: [
          'python generate_ledger.py --image-dir /mnt/hdd/grid_runs --output global_exclude_ledger.txt',
          'python generate_ledger.py --image-dir /mnt/hdd/grid_runs --output global_exclude_ledger.txt --substring North_America',
        ],
        options: [
          { option: '--image-dir', default: 'required', description: 'Base directory containing grid run folders with images.' },
          { option: '--output', default: 'global_exclude_ledger.txt', description: 'Ledger file to create or append to.' },
          { option: '--substring', default: 'unset', description: 'Only include images from folders whose path contains this string.' },
        ],
      },
    ],
  },
  {
    title: 'Checkpoint Maintenance',
    scripts: [
      {
        name: 'check_zst_health.py',
        description:
          'Tests all .zst files under the grid run directories using zstd -t. When --clear-completed is set, deletes the corresponding .completed_<sub_id> marker so the main script will re-process the affected sub-grid on the next run.',
        usage: 'python check_zst_health.py [options]',
        examples: [
          'python check_zst_health.py',
          'python check_zst_health.py --delete-all --clear-completed --ignore-recent 1.5',
        ],
        options: [
          { option: '-d / --delete-all', default: 'False', description: 'Delete all corrupted files without prompting.' },
          { option: '-s / --substring', default: 'unset', description: 'Only check files whose path contains this string.' },
          { option: '-i / --ignore-recent', default: '0', description: 'Skip files modified within the last N hours.' },
          { option: '-c / --clear-completed', default: 'False', description: 'Delete the corresponding .completed_<sub_id> marker on corruption.' },
          { option: '-w / --workers', default: 'CPU count', description: 'Concurrent workers.' },
        ],
      },
    ],
  },
  {
    title: 'Audit & Repair',
    scripts: [
      {
        name: 'audit_markers.py',
        description:
          'Scans grid_runs/ for orphaned .completed_* resume markers — markers whose corresponding metadata_checkpoint_*.jsonl.zst or animal_detections_checkpoint_*.jsonl.zst files are missing. Orphaned markers would otherwise convince the main script that a sub-grid finished successfully, causing it to be skipped silently on the next run. Any orphaned markers found are deleted automatically.',
        usage: 'python audit_markers.py',
        examples: ['python audit_markers.py'],
      },
      {
        name: 'audit_silent_skips.py',
        description:
          'A multiprocessed auditor that detects silent skips — sub-grids marked as .completed_ but whose checkpoint files contain fewer records than expected. For each completed sub-grid it loads the topology checkpoint to get the expected image count, then counts lines in the metadata and animal-detection checkpoints. If either count falls short, the .completed_ marker is deleted to force a backfill rerun.',
        usage: 'python audit_silent_skips.py [options]',
        examples: [
          'python audit_silent_skips.py',
          'python audit_silent_skips.py --dry-run --substring North_America',
        ],
        options: [
          { option: '--workers', default: 'all cores', description: 'Parallel CPU workers for checkpoint parsing.' },
          { option: '--dry-run', default: 'False', description: 'Report discrepancies without deleting markers.' },
          { option: '--substring', default: 'unset', description: 'Only audit sub-grids whose path contains this string.' },
        ],
      },
      {
        name: 'generate_rerun_commands.py',
        description:
          'Reads a grid CSV and checks every region directory for missing .completed_* or .empty_* markers. For each region with incomplete sub-grids, it generates a ready-to-run batch_chunks_mp_api.py command using --row-index and --sub-indices to target only the missing cells.',
        usage: 'python generate_rerun_commands.py <regions.csv> [options]',
        examples: [
          'python generate_rerun_commands.py regions.csv',
          'python generate_rerun_commands.py regions.csv --substring "South America" --output-script rerun_sa.sh',
        ],
        options: [
          { option: '--parent-dir', default: 'grid_runs', description: 'Directory containing region output folders.' },
          { option: '--image-dir', default: 'unset', description: 'Optional separate image directory to include in the generated commands.' },
          { option: '--substring', default: 'unset', description: 'Filter to rows whose region name contains this string.' },
          { option: '--sub-grid-step', default: '1.0', description: 'Must match the --sub-grid-step used in the main script.' },
          { option: '--output-script', default: 'run_missing.sh', description: 'Name of the generated bash script.' },
        ],
      },
    ],
  },
  {
    title: 'Coverage Audit & Image Repair',
    scripts: [
      {
        name: 'tools/coverage/validate_missing_sample.py',
        description:
          'Samples the in-scope missing set (every k-th row, so it spans all cells) and probes /detections to estimate how many images are still live vs gone and what fraction are ground animals, then extrapolates to the full set so you can size the backfill download volume before committing to it.',
        usage: 'python tools/coverage/validate_missing_sample.py --inscope <dir> [--region <name>] [-n <sample>]',
        examples: [
          'python tools/coverage/validate_missing_sample.py --inscope coverage_missing_inscope --region Europe -n 2000',
        ],
      },
      {
        name: 'tools/repair/diagnose_images.py',
        description:
          'Compares ground_animals_* parquet image_ids against the .jpg files on disk for each region, reporting orphaned images (file exists, not in any parquet) and missing downloads (in parquet, no file). Run this first to understand a region whose progress shows over 100%.',
        usage: 'python tools/repair/diagnose_images.py <grid.csv> --dirs <dir> [<dir> ...] [--region <name>]',
        examples: [
          'python tools/repair/diagnose_images.py original_global_grid_5deg.csv --dirs grid_runs --region "South Asia"',
        ],
      },
      {
        name: 'tools/repair/fetch_missing_images.py',
        description:
          'Finds image_ids that are in a manifest but have no jpg on disk, then downloads them, trying the stored thumb_original_url first and, on failure, fetching a fresh signed URL from the Graph API (rotating across all MLY_KEY* tokens). Images the API reports gone are recorded as permanently dead. Subcommands: scan, download, all.',
        usage: 'python tools/repair/fetch_missing_images.py {scan|download|all} <grid.csv> --dirs <dir> [...]',
        examples: [
          'python tools/repair/fetch_missing_images.py all original_global_grid_5deg.csv --dirs grid_runs --proxy-file proxies.txt -w 24',
        ],
      },
      {
        name: 'tools/repair/rebuild_manifest_from_images.py',
        description:
          'Treats images as the source of truth (the over-100% case): for every orphan image_id it re-queries Mapillary detections, keeps only those still classified animal--ground-animal, and writes reconstructed ground_animals_*_recovered_*.parquet rows so the manifest count rises to match the images on disk.',
        usage: 'python tools/repair/rebuild_manifest_from_images.py {scan|rebuild|all} <grid.csv> --dirs <dir> [...]',
        examples: [
          'python tools/repair/rebuild_manifest_from_images.py all original_global_grid_5deg.csv --dirs grid_runs --region "South Asia" -w 24',
        ],
      },
      {
        name: 'tools/repair/drop_dead_from_manifest.py',
        description:
          'Removes permanently-dead image_ids (listed in data/dead_images.txt) from every ground_animals_* parquet so they stop inflating the missing gap. Archives the removed rows first (restorable), rewrites affected files atomically, and is idempotent. Dry-run by default; pass --execute to write.',
        usage: 'python tools/repair/drop_dead_from_manifest.py --dirs <dir> [<dir> ...] [--execute]',
        examples: [
          'python tools/repair/drop_dead_from_manifest.py --dirs grid_runs /mnt/hdd/grid_runs',
          'python tools/repair/drop_dead_from_manifest.py --dirs grid_runs /mnt/hdd/grid_runs --execute',
        ],
      },
      {
        name: 'tools/repair/deduplicate_parquets.py',
        description:
          'Multiprocessed pass over every *.parquet that removes duplicate image_id rows (keeping the first occurrence) and rewrites only files that actually had duplicates, with zstd compression matching the main script.',
        usage: 'python tools/repair/deduplicate_parquets.py --parent-dir <dir> [--substring <s>]',
        examples: [
          'python tools/repair/deduplicate_parquets.py --parent-dir grid_runs --substring North_America',
        ],
      },
    ],
  },
  {
    title: 'Visualization',
    scripts: [
      {
        name: 'visualize_region_tiles.py',
        description:
          'Generates a static map image showing mercantile tiles for a region, colored green (land) or red (water). Saves the PNG into the region\'s output folder. Requires a folder name in the format Name_SWLon_SWLat_NELon_NELat.',
        usage: 'python visualize_region_tiles.py "<folder_name>" [options]',
        examples: [
          'python visualize_region_tiles.py "Sample_Region_-74.1_40.6_-73.7_40.9"',
          'python visualize_region_tiles.py "Sample_Region_-74.1_40.6_-73.7_40.9" --zoom 14 --parent_dir grid_runs',
        ],
      },
    ],
  },
  {
    title: 'Web Browser',
    scripts: [
      {
        name: 'browse.py',
        description:
          'A single-file Flask web application for interactively browsing pipeline output data stored under one or more --dirs directories. Provides region sidebar, location search, data type tabs, paginated listings, image lightbox, interactive Leaflet maps, and download buttons.',
        usage: 'python browse.py --dirs <dir> [<dir> ...] [options]',
        examples: [
          'python browse.py --dirs grid_runs /mnt/hdd/grid_runs',
          'python browse.py --dirs grid_runs --port 8080 --host 0.0.0.0',
        ],
        options: [
          { option: '--dirs', default: 'grid_runs', description: 'One or more base directories to scan for region folders.' },
          { option: '--host', default: '127.0.0.1', description: 'Address to bind the server to.' },
          { option: '--port', default: '5000', description: 'Port to listen on.' },
        ],
      },
    ],
  },
]

export default function HelperScripts() {
  const mainRef = useRef<HTMLElement>(null)
  const cardsRef = useRef<HTMLDivElement[]>([])

  useEffect(() => {
    const main = mainRef.current
    if (!main) return

    const cards = cardsRef.current.filter(Boolean)
    const ctx = gsap.context(() => {
      gsap.from(cards, {
        y: 20,
        opacity: 0,
        stagger: 0.05,
        duration: 0.4,
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
              Utilities
            </span>
          </div>
          <h1 className="text-[clamp(2rem,4vw,3.5rem)] font-normal leading-[1.15] tracking-tight text-[#f8f9fa]">
            Helper Scripts
          </h1>
          <p className="mt-5 max-w-[720px] text-[15px] leading-[1.65] text-[#adb5bd]">
            Standalone utilities that prepare grids, track progress, audit checkpoints, manage exclude
            ledgers, and visualize pipeline output. All scripts live in the repository root and can be
            run independently of the main pipeline. Legacy gzip-only scripts are omitted because the
            pipeline now uses zstd checkpoints.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              to="/cli-reference"
              className="group inline-flex items-center gap-2 rounded-md border border-[#495057] bg-transparent px-5 py-2.5 text-[14px] font-medium text-[#f8f9fa] transition-colors duration-200 hover:border-[#adb5bd]"
            >
              CLI Reference
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

        <section className="mx-auto max-w-[1000px] px-6">
          {categories.map((category, catIndex) => (
            <div key={category.title} className="mb-16">
              <div className="mb-6 flex items-center gap-3">
                <Wrench size={20} className="text-[#e8a645]" />
                <h2 className="text-[clamp(1.3rem,2.2vw,1.6rem)] font-normal text-[#f8f9fa]">
                  {category.title}
                </h2>
              </div>

              <div className="space-y-6">
                {category.scripts.map((script, scriptIndex) => {
                  const refIndex = catIndex * 20 + scriptIndex
                  return (
                    <div
                      key={script.name}
                      ref={(el) => { if (el) cardsRef.current[refIndex] = el }}
                      className="rounded-lg border border-[rgba(73,80,87,0.5)] bg-[#212529] p-6 transition-colors duration-200 hover:border-[#495057]"
                    >
                      <div className="font-mono text-[15px] font-medium text-[#e8a645]">{script.name}</div>
                      <p className="mt-2 text-[14px] leading-[1.65] text-[#adb5bd]">{script.description}</p>

                      <div className="mt-5">
                        <h4 className="mb-1.5 text-[12px] font-medium uppercase tracking-wider text-[#6c757d]">
                          Usage
                        </h4>
                        <div className="rounded-md bg-[#1e1e1e] border border-[rgba(73,80,87,0.3)] p-3">
                          <code className="font-mono text-[13px] text-[#d4d4d4]">{script.usage}</code>
                        </div>
                      </div>

                      {script.examples && (
                        <div className="mt-4">
                          <h4 className="mb-1.5 text-[12px] font-medium uppercase tracking-wider text-[#6c757d]">
                            Examples
                          </h4>
                          <div className="space-y-2">
                            {script.examples.map((ex, i) => (
                              <div
                                key={i}
                                className="rounded-md bg-[#1e1e1e] border border-[rgba(73,80,87,0.3)] p-3"
                              >
                                <pre className="overflow-x-auto font-mono text-[13px] leading-[1.6] text-[#d4d4d4]">
                                  <code>{ex}</code>
                                </pre>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {script.options && (
                        <div className="mt-4">
                          <h4 className="mb-2 text-[12px] font-medium uppercase tracking-wider text-[#6c757d]">
                            Options
                          </h4>
                          <div className="overflow-x-auto rounded-md border border-[rgba(73,80,87,0.3)]">
                            <table className="w-full text-left">
                              <tbody>
                                {script.options.map((opt, i) => (
                                  <tr
                                    key={opt.option}
                                    className={`${
                                      i % 2 === 0 ? 'bg-[#1e1e1e]' : 'bg-[#25282c]'
                                    } border-t border-[rgba(73,80,87,0.3)]`}
                                  >
                                    <td className="whitespace-nowrap px-3 py-2 font-mono text-[12px] text-[#e8a645]">
                                      {opt.option}
                                    </td>
                                    <td className="whitespace-nowrap px-3 py-2 font-mono text-[12px] text-[#adb5bd]">
                                      {opt.default}
                                    </td>
                                    <td className="px-3 py-2 text-[13px] text-[#adb5bd]">
                                      {opt.description}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </section>
      </main>

      <FooterSection />
    </div>
  )
}
