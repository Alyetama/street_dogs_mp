import { useRef, useEffect } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { Globe, RefreshCw, Grid3X3, Server, FlaskConical, Zap } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

const features = [
  {
    icon: Globe,
    title: 'Continental Scale',
    description:
      'Process thousands of geographic regions in parallel using multiprocessing. Designed for planet-scale data extraction.',
  },
  {
    icon: RefreshCw,
    title: 'Resumable Pipeline',
    description:
      'zstd-compressed checkpoints and completion markers let you resume interrupted runs without losing progress.',
  },
  {
    icon: Grid3X3,
    title: 'Mercantile Grid',
    description:
      'Splits regions into Mapillary tiles at configurable zoom levels. Filters to land-only tiles for efficiency.',
  },
  {
    icon: Server,
    title: 'SLURM Cluster Support',
    description:
      'Built-in SLURM array job support for HPC environments. Submit one region per task across hundreds of nodes.',
  },
  {
    icon: FlaskConical,
    title: 'Web Browser',
    description:
      'browse.py provides a Flask-based interactive browser with maps, image lightbox, and location search.',
  },
  {
    icon: Zap,
    title: 'Parallel Downloads',
    description:
      'Threaded image downloads with configurable workers, proxy support, and SSD write buffering for slow HDDs.',
  },
]

export default function FeaturesSection() {
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
    <section ref={sectionRef} className="bg-[#212529] py-20">
      <div className="mx-auto max-w-[1200px] px-6">
        <h2 className="relative mb-12 text-[clamp(1.8rem,3vw,2.5rem)] font-normal text-[#f8f9fa]">
          Key Features
          <span className="absolute -bottom-3 left-0 block h-[2px] w-10 bg-[#e8a645]" />
        </h2>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, i) => {
            const Icon = feature.icon
            return (
              <div
                key={feature.title}
                ref={(el) => { if (el) cardsRef.current[i] = el }}
                className="rounded-lg border border-[rgba(73,80,87,0.5)] bg-[#2c3034] p-8 transition-colors duration-200 hover:border-[#495057]"
              >
                <Icon size={40} className="text-[#e8a645]" strokeWidth={1.5} />
                <h3 className="mt-4 text-[18px] font-medium text-[#f8f9fa]">{feature.title}</h3>
                <p className="mt-2 text-[14px] leading-[1.6] text-[#adb5bd]">{feature.description}</p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
