import { useRef, useEffect } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

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

export default function CLIOptionsSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const rowsRef = useRef<HTMLTableRowElement[]>([])

  useEffect(() => {
    const section = sectionRef.current
    if (!section) return

    const rows = rowsRef.current.filter(Boolean)
    const ctx = gsap.context(() => {
      gsap.from(rows, {
        y: 15,
        opacity: 0,
        stagger: 0.05,
        duration: 0.4,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: section,
          start: 'top 80%',
        },
      })
    }, section)

    return () => ctx.revert()
  }, [])

  return (
    <section id="cli-options" ref={sectionRef} className="bg-[#1a1a1a] py-20">
      <div className="mx-auto max-w-[1000px] px-6">
        <h2 className="relative mb-12 text-[clamp(1.8rem,3vw,2.5rem)] font-normal text-[#f8f9fa]">
          CLI Options
          <span className="absolute -bottom-3 left-0 block h-[2px] w-10 bg-[#e8a645]" />
        </h2>

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
                  ref={(el) => { if (el) rowsRef.current[i] = el }}
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
            python batch_chunks_mp_api.py --help
          </code>
        </p>
      </div>
    </section>
  )
}
