import { Link } from 'react-router'
import GlobeCanvas from './GlobeCanvas'

export default function HeroSection() {
  const scrollTo = (id: string) => {
    const el = document.querySelector(id)
    if (el) el.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section className="relative min-h-[100dvh] flex items-center overflow-hidden bg-[#1a1a1a]">
      <GlobeCanvas />

      <div className="relative z-10 mx-auto w-full max-w-[1200px] px-6 md:px-10 lg:px-20">
        <div className="max-w-[560px]">
          {/* Tag */}
          <div className="mb-6 inline-flex items-center rounded-[20px] border border-[rgba(232,166,69,0.3)] px-3 py-1">
            <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-[#e8a645]">
              Geographic Ingestion Pipeline
            </span>
          </div>

          {/* Heading */}
          <h1 className="text-[clamp(2.5rem,5vw,4.5rem)] font-normal leading-[1.1] tracking-tight">
            <span className="text-[#f8f9fa]">Find </span>
            <span className="text-[#e8a645]">street dogs</span>
            <br />
            <span className="text-[#f8f9fa]">across the planet,</span>
            <br />
            <span className="text-[#f8f9fa]">one tile at a time.</span>
          </h1>

          {/* Subtitle */}
          <p className="mt-6 max-w-[520px] text-[15px] leading-[1.65] text-[#adb5bd]">
            A resumable, multiprocessing Mapillary pipeline. Scan geographic grid cells,
            extract image metadata, isolate{' '}
            <code className="rounded bg-[#2c3034] px-1.5 py-0.5 text-[13px] text-[#e8a645]">
              animal--ground-animal
            </code>{' '}
            detections, and download the matching imagery — at continental scale.
          </p>

          {/* CTAs */}
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <button
              onClick={() => scrollTo('#quick-start')}
              className="rounded-md bg-[#e8a645] px-6 py-2.5 text-[14px] font-medium text-[#1a1a1a] transition-colors duration-200 hover:bg-[#f0b85c]"
            >
              Get Started
            </button>
            <Link
              to="/cli-reference"
              className="rounded-md border border-[#495057] bg-transparent px-6 py-2.5 text-[14px] font-medium text-[#f8f9fa] transition-colors duration-200 hover:border-[#adb5bd]"
            >
              CLI Reference
            </Link>
          </div>

          {/* Stats */}
          <div className="mt-12 flex flex-wrap items-end gap-8">
            <div>
              <div className="flex h-9 items-end">
                <div className="text-[36px] font-normal leading-none text-[#e8a645]">6</div>
              </div>
              <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.06em] text-[#6c757d]">
                Phase Workflow
              </div>
            </div>
            <div>
              <div className="flex h-9 items-end">
                <div className="font-mono text-[28px] font-normal leading-none text-[#e8a645]">zstd</div>
              </div>
              <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.06em] text-[#6c757d]">
                Checkpoints
              </div>
            </div>
            <div>
              <div className="flex h-9 items-end">
                <div className="text-[36px] font-normal leading-none text-[#e8a645]">&infin;</div>
              </div>
              <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.06em] text-[#6c757d]">
                Resumable
              </div>
            </div>
            <div>
              <div className="flex h-9 items-end">
                <div className="font-mono text-[28px] font-normal leading-none text-[#e8a645]">.parquet</div>
              </div>
              <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.06em] text-[#6c757d]">
                Data Files
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
