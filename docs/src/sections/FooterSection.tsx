export default function FooterSection() {
  return (
    <footer className="border-t border-[rgba(73,80,87,0.5)] bg-[#1a1a1a] py-10">
      <div className="mx-auto flex max-w-[1200px] flex-col items-center justify-between gap-4 px-6 md:flex-row">
        {/* Left */}
        <div className="flex items-center gap-3">
          <span className="text-[14px] font-medium text-[#f8f9fa]">street_dogs_mp</span>
          <span className="rounded bg-[#2c3034] px-2 py-0.5 text-[11px] text-[#6c757d]">v3.0</span>
        </div>

        {/* Center */}
        <a
          href="https://github.com/Alyetama/street_dogs_mp/blob/main/LICENSE"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[13px] text-[#6c757d] transition-colors duration-200 hover:text-[#adb5bd]"
        >
          MIT License
        </a>

        {/* Right */}
        <a
          href="https://github.com/Alyetama/street_dogs_mp"
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-[13px] text-[#adb5bd] transition-colors duration-200 hover:text-[#e8a645]"
        >
          Alyetama/street_dogs_mp
        </a>
      </div>
    </footer>
  )
}
