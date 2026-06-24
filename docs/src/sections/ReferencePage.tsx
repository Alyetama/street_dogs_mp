import { useRef, useEffect } from 'react'
import { Link } from 'react-router'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import Navigation from './Navigation'
import FooterSection from './FooterSection'
import { Terminal, FileOutput, ListTree, ArrowRight } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

export type RefOption = { option: string; default: string; description: string }
export type RefCommand = { name: string; description: string }
export type RefExample = { title: string; code: string }
export type RefOutput = { file: string; description: string }

export type ReferencePageProps = {
  badge: string
  title: string
  lead: string
  primaryLink?: { label: string; to: string }
  subcommands?: { intro?: string; items: RefCommand[] }
  options: RefOption[]
  optionsHelp?: string
  examples: RefExample[]
  outputIntro?: React.ReactNode
  outputs: RefOutput[]
}

const GITHUB_URL = 'https://github.com/Alyetama/street_dogs_mp'

const thClass =
  'px-4 py-3 text-[12px] font-medium uppercase tracking-wider text-[#e8a645]'
const sectionH2 =
  'text-[clamp(1.4rem,2.5vw,1.8rem)] font-normal text-[#f8f9fa]'

export default function ReferencePage({
  badge,
  title,
  lead,
  primaryLink,
  subcommands,
  options,
  optionsHelp,
  examples,
  outputIntro,
  outputs,
}: ReferencePageProps) {
  const mainRef = useRef<HTMLElement>(null)
  const cmdRef = useRef<HTMLDivElement>(null)
  const tableRef = useRef<HTMLDivElement>(null)
  const examplesRef = useRef<HTMLDivElement>(null)
  const outputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const main = mainRef.current
    if (!main) return
    const sections = [
      cmdRef.current,
      tableRef.current,
      examplesRef.current,
      outputRef.current,
    ].filter(Boolean)
    const ctx = gsap.context(() => {
      gsap.from(sections, {
        y: 25,
        opacity: 0,
        stagger: 0.12,
        duration: 0.5,
        ease: 'power2.out',
        scrollTrigger: { trigger: main, start: 'top 80%' },
      })
    }, main)
    return () => ctx.revert()
  }, [])

  return (
    <div className="min-h-screen bg-[#1a1a1a]">
      <Navigation />

      <main ref={mainRef} className="pt-24 pb-20">
        {/* Hero */}
        <section className="mx-auto max-w-[1000px] px-6 pb-10">
          <div className="mb-6 inline-flex items-center rounded-[20px] border border-[rgba(232,166,69,0.3)] px-3 py-1">
            <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-[#e8a645]">
              {badge}
            </span>
          </div>
          <h1 className="text-[clamp(2rem,4vw,3.5rem)] font-normal leading-[1.15] tracking-tight text-[#f8f9fa]">
            <span className="text-[#e8a645]">{title}</span>
          </h1>
          <p className="mt-5 max-w-[760px] text-[15px] leading-[1.65] text-[#adb5bd]">
            {lead}
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            {primaryLink && (
              <Link
                to={primaryLink.to}
                className="group inline-flex items-center gap-2 rounded-md border border-[#495057] bg-transparent px-5 py-2.5 text-[14px] font-medium text-[#f8f9fa] transition-colors duration-200 hover:border-[#adb5bd]"
              >
                {primaryLink.label}
                <ArrowRight
                  size={16}
                  className="transition-transform duration-200 group-hover:translate-x-1"
                />
              </Link>
            )}
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-md bg-[#e8a645] px-5 py-2.5 text-[14px] font-medium text-[#1a1a1a] transition-colors duration-200 hover:bg-[#f0b85c]"
            >
              View on GitHub
            </a>
          </div>
        </section>

        {/* Subcommands (optional) */}
        {subcommands && (
          <section ref={cmdRef} className="mx-auto max-w-[1000px] px-6 py-12">
            <div className="mb-8 flex items-center gap-3">
              <ListTree size={22} className="text-[#e8a645]" />
              <h2 className={sectionH2}>Subcommands</h2>
            </div>
            {subcommands.intro && (
              <p className="mb-6 text-[14px] leading-[1.65] text-[#adb5bd]">
                {subcommands.intro}
              </p>
            )}
            <div className="overflow-x-auto rounded-lg border border-[#495057]">
              <table className="w-full text-left">
                <thead>
                  <tr className="bg-[#2c3034]">
                    <th className={thClass}>Command</th>
                    <th className={thClass}>Purpose</th>
                  </tr>
                </thead>
                <tbody>
                  {subcommands.items.map((c, i) => (
                    <tr
                      key={c.name}
                      className={`${i % 2 === 0 ? 'bg-[#212529]' : 'bg-[#1e1e1e]'} border-t border-[rgba(73,80,87,0.3)]`}
                    >
                      <td className="whitespace-nowrap px-4 py-3 font-mono text-[13px] text-[#e8a645]">
                        {c.name}
                      </td>
                      <td className="px-4 py-3 text-[14px] text-[#adb5bd]">
                        {c.description}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Options */}
        <section ref={tableRef} className="mx-auto max-w-[1000px] px-6 py-12">
          <div className="mb-8 flex items-center gap-3">
            <Terminal size={22} className="text-[#e8a645]" />
            <h2 className={sectionH2}>CLI Options</h2>
          </div>
          <div className="overflow-x-auto rounded-lg border border-[#495057]">
            <table className="w-full text-left">
              <thead>
                <tr className="bg-[#2c3034]">
                  <th className={thClass}>Option</th>
                  <th className={thClass}>Default</th>
                  <th className={thClass}>Description</th>
                </tr>
              </thead>
              <tbody>
                {options.map((opt, i) => (
                  <tr
                    key={opt.option}
                    className={`${i % 2 === 0 ? 'bg-[#212529]' : 'bg-[#1e1e1e]'} border-t border-[rgba(73,80,87,0.3)]`}
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
          {optionsHelp && (
            <p className="mt-4 text-[13px] text-[#6c757d]">
              For the complete reference, run:{' '}
              <code className="rounded bg-[#2c3034] px-1.5 py-0.5 font-mono text-[12px] text-[#e8a645]">
                {optionsHelp}
              </code>
            </p>
          )}
        </section>

        {/* Usage examples */}
        <section ref={examplesRef} className="mx-auto max-w-[1000px] px-6 py-12">
          <h2 className={`mb-8 ${sectionH2}`}>Usage Examples</h2>
          <div className="space-y-6">
            {examples.map((ex) => (
              <div key={ex.title}>
                <h3 className="mb-2 text-[14px] font-medium text-[#f8f9fa]">
                  {ex.title}
                </h3>
                <div className="rounded-lg bg-[#1e1e1e] border border-[rgba(73,80,87,0.5)] p-4">
                  <pre className="overflow-x-auto font-mono text-[13px] leading-[1.7] text-[#d4d4d4]">
                    <code>{ex.code}</code>
                  </pre>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Outputs */}
        <section ref={outputRef} className="mx-auto max-w-[1000px] px-6 py-12">
          <div className="mb-8 flex items-center gap-3">
            <FileOutput size={22} className="text-[#e8a645]" />
            <h2 className={sectionH2}>Outputs</h2>
          </div>
          {outputIntro && (
            <div className="mb-6 text-[14px] leading-[1.65] text-[#adb5bd]">
              {outputIntro}
            </div>
          )}
          <div className="overflow-x-auto rounded-lg border border-[#495057]">
            <table className="w-full text-left">
              <thead>
                <tr className="bg-[#2c3034]">
                  <th className={thClass}>File / Directory</th>
                  <th className={thClass}>Description</th>
                </tr>
              </thead>
              <tbody>
                {outputs.map((out, i) => (
                  <tr
                    key={out.file}
                    className={`${i % 2 === 0 ? 'bg-[#212529]' : 'bg-[#1e1e1e]'} border-t border-[rgba(73,80,87,0.3)]`}
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
