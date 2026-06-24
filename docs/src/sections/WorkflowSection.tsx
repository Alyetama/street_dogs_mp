import { useRef, useEffect } from 'react'
import { Link } from 'react-router'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { ArrowRight } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

const stages = [
  {
    number: 1,
    title: 'Extract',
    href: '/cli-reference',
    script: 'batch_chunks_mp_api.py',
    description: 'Scan grid cells and pull every ground-animal image.',
    steps: [
      'Tile each region (land tiles only)',
      'Query sequences → image IDs',
      'Fetch metadata + detections',
      'Filter animal--ground-animal',
      'Write Parquet + download jpgs',
    ],
  },
  {
    number: 2,
    title: 'Audit',
    href: '/coverage-audit',
    script: 'coverage_audit.py',
    description: 'Verify nothing was missed, using Mapillary vector tiles.',
    steps: [
      'Enumerate every z14 land tile',
      'Retry any failed tiles',
      'Diff coverage vs extracted data',
      'Date-filter the in-scope gap',
    ],
  },
  {
    number: 3,
    title: 'Backfill',
    href: '/backfill',
    script: 'backfill_missing.py',
    description: 'Fetch and download the in-scope missing images.',
    steps: [
      'Read the in-scope missing set',
      'Metadata + detections per image',
      'Write backfill Parquet chunks',
      'Download ground-animal jpgs',
    ],
  },
]

export default function WorkflowSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const cardsRef = useRef<HTMLAnchorElement[]>([])

  useEffect(() => {
    const section = sectionRef.current
    if (!section) return

    const cards = cardsRef.current.filter(Boolean)
    const ctx = gsap.context(() => {
      gsap.from(cards, {
        y: 30,
        opacity: 0,
        stagger: 0.12,
        duration: 0.6,
        ease: 'power3.out',
        scrollTrigger: {
          trigger: section,
          start: 'top 80%',
        },
      })
    }, section)

    return () => ctx.revert()
  }, [])

  return (
    <section
      id="workflow"
      ref={sectionRef}
      className="bg-[#212529] py-20"
      style={{ backgroundImage: 'url(/workflow-bg.jpg)', backgroundSize: 'cover', backgroundPosition: 'center', backgroundBlendMode: 'overlay' }}
    >
      <div className="mx-auto max-w-[1200px] px-6">
        <h2 className="relative mb-4 text-[clamp(1.8rem,3vw,2.5rem)] font-normal text-[#f8f9fa]">
          Pipeline Workflow
          <span className="absolute -bottom-3 left-0 block h-[2px] w-10 bg-[#e8a645]" />
        </h2>
        <p className="mb-12 mt-6 max-w-[680px] text-[15px] leading-[1.65] text-[#adb5bd]">
          Three independent stages, each a standalone script. Extract harvests
          the images, Audit proves the harvest is complete, and Backfill closes
          any gap. Click a stage for its full CLI reference.
        </p>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {stages.map((stage, i) => (
            <Link
              to={stage.href}
              key={stage.number}
              ref={(el) => { if (el) cardsRef.current[i] = el }}
              className="group relative flex flex-col overflow-hidden rounded-lg border border-[rgba(73,80,87,0.5)] bg-[#2c3034] p-6 transition-colors duration-200 hover:border-[rgba(232,166,69,0.5)]"
            >
              {/* Animated border effect */}
              <div
                className="absolute inset-[-1px] rounded-[9px] opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                style={{
                  background: 'conic-gradient(from 0deg, transparent 0%, rgba(232,166,69,0.4) 25%, transparent 50%)',
                  animation: 'rotate 4s linear infinite',
                }}
              />
              <div className="relative z-10 flex h-full flex-col">
                <div className="flex items-center gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[rgba(232,166,69,0.15)] text-[14px] font-medium text-[#e8a645]">
                    {stage.number}
                  </div>
                  <h3 className="text-[18px] font-medium text-[#f8f9fa]">{stage.title}</h3>
                </div>
                <code className="mt-3 block font-mono text-[12px] text-[#e8a645]">{stage.script}</code>
                <p className="mt-2 text-[14px] leading-[1.6] text-[#adb5bd]">{stage.description}</p>
                <ul className="mt-4 space-y-1.5">
                  {stage.steps.map((step) => (
                    <li key={step} className="flex items-start gap-2 text-[13px] leading-[1.5] text-[#adb5bd]">
                      <span className="mt-[7px] block h-1 w-1 shrink-0 rounded-full bg-[#e8a645]" />
                      {step}
                    </li>
                  ))}
                </ul>
                <span className="mt-5 inline-flex items-center gap-1.5 text-[13px] font-medium text-[#e8a645]">
                  View reference
                  <ArrowRight size={14} className="transition-transform duration-200 group-hover:translate-x-1" />
                </span>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  )
}
