import { useState, useRef, useEffect } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const tabs = [
  { id: 'install', label: 'Install' },
  { id: 'token', label: 'Token Setup' },
  { id: 'run', label: 'Run' },
  { id: 'audit', label: 'Audit + Backfill' },
  { id: 'output', label: 'Output' },
]

const tabContent: Record<string, { code?: string; html?: string; description?: string }> = {
  install: {
    code: `# Install dependencies
python -m pip install -r requirements.txt

# Key dependencies:
# mercantile, polars, requests, python-dotenv
# global-land-mask, tqdm, contextily
# matplotlib, orjson, piexif, geopandas`,
  },
  token: {
    code: `# Create a .env file in the project root
MLY_KEY=your_mapillary_access_token

# You can also define numbered tokens:
MLY_KEY_1=your_first_token
MLY_KEY_2=your_second_token

# Select a numbered token with --token:
python batch_chunks_mp_api.py regions.csv --token 1`,
  },
  run: {
    code: `# Run all regions in a CSV
python batch_chunks_mp_api.py regions.csv

# Run one specific row by index
python batch_chunks_mp_api.py regions.csv --row-index 0

# Fetch metadata without downloading images
python batch_chunks_mp_api.py regions.csv --no-download-images

# Download images later from existing Parquet files
python batch_chunks_mp_api.py regions.csv --download-only

# Use a separate disk for images
python batch_chunks_mp_api.py regions.csv --image-dir /path/to/storage

# Use SSD as write buffer for slow HDD
python batch_chunks_mp_api.py regions.csv \\
  --image-dir /mnt/hdd/images \\
  --temp-dir /tmp/ssd_spool

# Process specific sub-grid indices
python batch_chunks_mp_api.py regions.csv --sub-indices 4,12,15`,
  },
  audit: {
    code: `# Stage 2 — audit coverage (enumerate -> retry -> diff -> datefilter)
python coverage_audit.py audit original_global_grid_5deg.csv \\
  --dirs grid_runs /mnt/hdd/grid_runs --region Europe --wait

# List the in-scope missing set (writes coverage_missing_inscope/)
python coverage_audit.py datefilter --cutoff 2026-05-31

# Stage 3 — backfill the missing set (parquets + images)
python backfill_missing.py \\
  --inscope coverage_missing_inscope --region Europe \\
  --out-dir /path/to/grid_runs --image-dir /path/to/images \\
  --processes 3

# Download images for an already-backfilled region
python backfill_missing.py --download-only \\
  --out-dir /path/to/grid_runs --image-dir /path/to/images`,
  },
  output: {
    html: `<div class="text-[#adb5bd] text-[13px] leading-[1.7] font-mono">
<span class="text-[#6a9955]"># Output directory structure</span>
<pre>
grid_runs/
<span class="text-[#569cd6]">├──</span> &lt;safe_region_id&gt;/
<span class="text-[#569cd6]">│   ├──</span> topology_checkpoint_&lt;sub&gt;.json.zst
<span class="text-[#569cd6]">│   ├──</span> metadata_checkpoint_&lt;sub&gt;.jsonl.zst
<span class="text-[#569cd6]">│   ├──</span> animal_detections_checkpoint_&lt;sub&gt;.jsonl.zst
<span class="text-[#569cd6]">│   ├──</span> ground_animals_&lt;sub&gt;.json.zst
<span class="text-[#569cd6]">│   ├──</span> all_data_&lt;region&gt;_&lt;part&gt;.parquet
<span class="text-[#569cd6]">│   ├──</span> ground_animals_&lt;region&gt;_&lt;part&gt;.parquet
<span class="text-[#569cd6]">│   ├──</span> ground_animal_images/
<span class="text-[#569cd6]">│   │   └──</span> *.jpg
<span class="text-[#569cd6]">│   ├──</span> validated_images_&lt;region&gt;.txt
<span class="text-[#569cd6]">│   ├──</span> failed_downloads_&lt;region&gt;.txt
<span class="text-[#569cd6]">│   ├──</span> covered_countries.txt
<span class="text-[#569cd6]">│   ├──</span> .completed_&lt;sub&gt;
<span class="text-[#569cd6]">│   └──</span> .empty_&lt;sub&gt;
</pre>
</div>`,
  },
}

export default function QuickStartSection() {
  const [activeTab, setActiveTab] = useState('install')
  const sectionRef = useRef<HTMLElement>(null)
  const codeRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const section = sectionRef.current
    if (!section) return

    const ctx = gsap.context(() => {
      gsap.from(codeRef.current, {
        y: 20,
        opacity: 0,
        duration: 0.5,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: section,
          start: 'top 75%',
        },
      })
    }, section)

    return () => ctx.revert()
  }, [])

  const content = tabContent[activeTab]

  return (
    <section id="quick-start" ref={sectionRef} className="bg-[#1a1a1a] py-20">
      <div className="mx-auto max-w-[900px] px-6">
        <h2 className="relative mb-12 text-[clamp(1.8rem,3vw,2.5rem)] font-normal text-[#f8f9fa]">
          Quick Start
          <span className="absolute -bottom-3 left-0 block h-[2px] w-10 bg-[#e8a645]" />
        </h2>

        {/* Tab bar */}
        <div className="mb-0 flex border-b border-[#495057]">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 text-[14px] font-normal transition-colors duration-200 ${
                activeTab === tab.id
                  ? 'border-b-2 border-[#e8a645] text-[#f8f9fa]'
                  : 'text-[#6c757d] hover:text-[#adb5bd]'
              }`}
              role="tab"
              aria-selected={activeTab === tab.id}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Code block */}
        <div
          ref={codeRef}
          className="rounded-b-lg rounded-tr-lg bg-[#1e1e1e] p-5"
          role="tabpanel"
          aria-label={`${tabs.find(t => t.id === activeTab)?.label} code example`}
        >
          {content.code && (
            <div className="font-mono text-[13px] leading-[1.7]">
              {content.code.split('\n').map((line, i) => {
                if (line.trim().startsWith('#')) {
                  return (
                    <div key={i} className="syntax-comment">
                      {line}
                    </div>
                  )
                }
                // Highlight 'python' keyword
                if (line.includes('python ') && !line.includes('#')) {
                  const idx = line.indexOf('python')
                  return (
                    <div key={i} className="text-[#d4d4d4]">
                      {line.slice(0, idx)}
                      <span className="syntax-keyword">python</span>
                      {line.slice(idx + 6)}
                    </div>
                  )
                }
                // Highlight strings in quotes
                const stringMatches: { start: number; end: number }[] = []
                let inString = false
                let stringStart = 0
                let quoteChar = ''
                for (let c = 0; c < line.length; c++) {
                  const ch = line[c]
                  if (!inString && (ch === '"' || ch === "'")) {
                    inString = true
                    stringStart = c
                    quoteChar = ch
                  } else if (inString && ch === quoteChar && line[c - 1] !== '\\') {
                    inString = false
                    stringMatches.push({ start: stringStart, end: c + 1 })
                  }
                }
                if (stringMatches.length > 0) {
                  const parts: React.ReactNode[] = []
                  let lastEnd = 0
                  stringMatches.forEach((match, mi) => {
                    if (match.start > lastEnd) {
                      parts.push(<span key={`t${mi}`} className="text-[#d4d4d4]">{line.slice(lastEnd, match.start)}</span>)
                    }
                    parts.push(<span key={`s${mi}`} className="syntax-string">{line.slice(match.start, match.end)}</span>)
                    lastEnd = match.end
                  })
                  if (lastEnd < line.length) {
                    parts.push(<span key="end" className="text-[#d4d4d4]">{line.slice(lastEnd)}</span>)
                  }
                  return <div key={i}>{parts}</div>
                }
                return <div key={i} className="text-[#d4d4d4]">{line}</div>
              })}
            </div>
          )}
          {content.html && (
            <div dangerouslySetInnerHTML={{ __html: content.html }} />
          )}
        </div>
      </div>
    </section>
  )
}
