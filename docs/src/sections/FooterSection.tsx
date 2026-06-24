import { Link } from 'react-router'
import { Github } from 'lucide-react'

const docLinks = [
  { label: 'Extract', to: '/cli-reference' },
  { label: 'Audit', to: '/coverage-audit' },
  { label: 'Backfill', to: '/backfill' },
  { label: 'Helper Scripts', to: '/helper-scripts' },
]

const projectLinks = [
  { label: 'GitHub', href: 'https://github.com/Alyetama/street_dogs_mp' },
  { label: 'Issues', href: 'https://github.com/Alyetama/street_dogs_mp/issues' },
  { label: 'MIT License', href: 'https://github.com/Alyetama/street_dogs_mp/blob/main/LICENSE' },
]

export default function FooterSection() {
  return (
    <footer className="border-t border-[rgba(73,80,87,0.5)] bg-[#1a1a1a]">
      <div className="mx-auto max-w-[1200px] px-6 py-14">
        <div className="grid grid-cols-1 gap-10 md:grid-cols-[1.5fr_1fr_1fr]">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-3">
              <svg width="26" height="26" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="14" cy="14" r="13" stroke="#e8a645" strokeWidth="1.5" fill="rgba(232,166,69,0.1)" />
                <ellipse cx="14" cy="18" rx="5" ry="4" fill="#e8a645" opacity="0.8" />
                <circle cx="10" cy="12" r="1.8" fill="#e8a645" />
                <circle cx="18" cy="12" r="1.8" fill="#e8a645" />
                <ellipse cx="14" cy="15" rx="2.5" ry="2" fill="#e8a645" opacity="0.6" />
              </svg>
              <span className="font-mono text-[15px] font-medium text-[#f8f9fa]">street_dogs_mp</span>
            </div>
            <p className="mt-4 max-w-[320px] text-[13px] leading-[1.6] text-[#6c757d]">
              A resumable, multiprocessing Mapillary pipeline for harvesting
              ground-animal imagery worldwide — extract, audit, and backfill.
            </p>
          </div>

          {/* Documentation */}
          <div>
            <h4 className="text-[11px] font-medium uppercase tracking-[0.08em] text-[#e8a645]">
              Documentation
            </h4>
            <ul className="mt-4 space-y-2.5">
              {docLinks.map((l) => (
                <li key={l.to}>
                  <Link
                    to={l.to}
                    className="text-[13px] text-[#adb5bd] transition-colors duration-200 hover:text-[#f8f9fa]"
                  >
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Project */}
          <div>
            <h4 className="text-[11px] font-medium uppercase tracking-[0.08em] text-[#e8a645]">
              Project
            </h4>
            <ul className="mt-4 space-y-2.5">
              {projectLinks.map((l) => (
                <li key={l.label}>
                  <a
                    href={l.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[13px] text-[#adb5bd] transition-colors duration-200 hover:text-[#f8f9fa]"
                  >
                    {l.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-[rgba(73,80,87,0.4)] pt-6 sm:flex-row">
          <span className="text-[13px] text-[#6c757d]">
            Released under the MIT License.
          </span>
          <a
            href="https://github.com/Alyetama/street_dogs_mp"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 font-mono text-[13px] text-[#adb5bd] transition-colors duration-200 hover:text-[#e8a645]"
          >
            <Github size={15} />
            Alyetama/street_dogs_mp
          </a>
        </div>
      </div>
    </footer>
  )
}
