import { useRef, useEffect } from 'react'
import { Link } from 'react-router'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { ArrowRight } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

const scripts = [
  {
    name: 'browse.py',
    description: 'Flask web app for browsing pipeline output with maps, images, and search.',
  },
  {
    name: 'split_regions.py',
    description: 'Splits a global grid CSV into individual per-region CSV files.',
  },
  {
    name: 'generate_countries.py',
    description: "Lists countries intersecting each region's bounding box.",
  },
  {
    name: 'progress_tracker.py',
    description: 'Rich-formatted progress table with completion percentages.',
  },
  {
    name: 'find_location_folder.py',
    description: 'Geocodes cities/countries and finds overlapping region folders.',
  },
  {
    name: 'scan_regions.py',
    description: 'Scans directories and recommends exact CLI flags for the main script.',
  },
  {
    name: 'generate_ledger.py',
    description: 'Builds exclude ledgers to skip already-downloaded images.',
  },
  {
    name: 'convert_to_zstd.py',
    description: 'Converts .gz checkpoints to .zst with byte-level verification.',
  },
  {
    name: 'check_zst_health.py',
    description: 'Tests all .zst files with optional marker cleanup.',
  },
  {
    name: 'audit_markers.py',
    description: 'Finds and removes orphaned completion markers.',
  },
  {
    name: 'audit_silent_skips.py',
    description: 'Detects silent skips by comparing checkpoint record counts.',
  },
  {
    name: 'generate_rerun_commands.py',
    description: 'Generates targeted rerun commands for incomplete sub-grids.',
  },
  {
    name: 'visualize_region_tiles.py',
    description: 'Generates static tile maps colored by land/water.',
  },
]

export default function HelperScriptsSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const cardsRef = useRef<HTMLDivElement[]>([])

  useEffect(() => {
    const section = sectionRef.current
    if (!section) return

    const cards = cardsRef.current.filter(Boolean)
    const ctx = gsap.context(() => {
      gsap.from(cards, {
        y: 15,
        opacity: 0,
        stagger: 0.04,
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
    <section id="helper-scripts" ref={sectionRef} className="bg-[#212529] py-20">
      <div className="mx-auto max-w-[1200px] px-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between mb-12">
          <h2 className="relative text-[clamp(1.8rem,3vw,2.5rem)] font-normal text-[#f8f9fa]">
            Helper Scripts
            <span className="absolute -bottom-3 left-0 block h-[2px] w-10 bg-[#e8a645]" />
          </h2>
          <Link
            to="/helper-scripts"
            className="group inline-flex items-center gap-2 text-[14px] font-medium text-[#e8a645] hover:text-[#f0b85c] transition-colors duration-200"
          >
            Full usage & examples
            <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-1" />
          </Link>
        </div>

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
          {scripts.map((script, i) => (
            <div
              key={script.name}
              ref={(el) => { if (el) cardsRef.current[i] = el }}
              className="rounded-md border border-[rgba(73,80,87,0.3)] bg-[#1e1e1e] px-5 py-4 transition-all duration-200 hover:border-[#495057] hover:bg-[#25282c]"
            >
              <div className="font-mono text-[14px] font-medium text-[#e8a645]">{script.name}</div>
              <div className="mt-1 text-[13px] text-[#adb5bd]">{script.description}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
