import { useRef, useEffect } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const workflowSteps = [
  {
    number: 1,
    title: 'Tile Generation',
    description: 'Splits the region into Mapillary/mercantile tiles and keeps land tiles only.',
  },
  {
    number: 2,
    title: 'Sequence Query',
    description: 'Queries Mapillary image sequences for each land tile.',
  },
  {
    number: 3,
    title: 'Image Expansion',
    description: 'Expands sequences into individual image IDs.',
  },
  {
    number: 4,
    title: 'Metadata Fetch',
    description: 'Fetches image metadata for all collected IDs.',
  },
  {
    number: 5,
    title: 'Detection Filter',
    description: 'Fetches detection records and filters for animal--ground-animal.',
  },
  {
    number: 6,
    title: 'Output & Download',
    description: 'Writes compressed checkpoints and Parquet outputs, then downloads matching images.',
  },
]

export default function WorkflowSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const cardsRef = useRef<HTMLDivElement[]>([])

  useEffect(() => {
    const section = sectionRef.current
    if (!section) return

    const cards = cardsRef.current.filter(Boolean)
    const ctx = gsap.context(() => {
      gsap.from(cards, {
        y: 30,
        opacity: 0,
        stagger: 0.1,
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
        <h2 className="relative mb-12 text-[clamp(1.8rem,3vw,2.5rem)] font-normal text-[#f8f9fa]">
          Pipeline Workflow
          <span className="absolute -bottom-3 left-0 block h-[2px] w-10 bg-[#e8a645]" />
        </h2>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {workflowSteps.map((step, i) => (
            <div
              key={step.number}
              ref={(el) => { if (el) cardsRef.current[i] = el }}
              className="group relative overflow-hidden rounded-lg border border-[rgba(73,80,87,0.5)] bg-[#2c3034] p-6 transition-colors duration-200"
            >
              {/* Animated border effect */}
              <div
                className="absolute inset-[-1px] rounded-[9px] opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                style={{
                  background: 'conic-gradient(from 0deg, transparent 0%, rgba(232,166,69,0.4) 25%, transparent 50%)',
                  animation: 'rotate 4s linear infinite',
                }}
              />
              <div className="relative z-10">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[rgba(232,166,69,0.15)] text-[14px] font-medium text-[#e8a645]">
                  {step.number}
                </div>
                <h3 className="mt-3 text-[16px] font-medium text-[#f8f9fa]">{step.title}</h3>
                <p className="mt-2 text-[14px] leading-[1.6] text-[#adb5bd]">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
