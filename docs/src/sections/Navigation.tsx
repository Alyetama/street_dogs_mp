import { useState, useEffect } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router'
import { Github, Menu, X } from 'lucide-react'

const pageLinks = [
  { label: 'Home', href: '/' },
  { label: 'CLI Reference', href: '/cli-reference' },
  { label: 'Helper Scripts', href: '/helper-scripts' },
]

const homeHashLinks = [
  { label: 'Quick Start', href: '#quick-start' },
  { label: 'Workflow', href: '#workflow' },
]

export default function Navigation() {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const isHome = location.pathname === '/'

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const handleHashClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault()
    setMobileOpen(false)
    if (!isHome) {
      navigate(`/${href}`)
    } else {
      const target = document.querySelector(href)
      if (target) {
        target.scrollIntoView({ behavior: 'smooth' })
      }
    }
  }

  const handleLogoClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault()
    setMobileOpen(false)
    if (!isHome) {
      navigate('/')
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }

  const linkBase =
    'text-[14px] font-normal transition-colors duration-200'
  const pageLinkClass = ({ isActive }: { isActive: boolean }) =>
    `${linkBase} ${isActive ? 'text-[#f8f9fa]' : 'text-[#adb5bd] hover:text-[#f8f9fa]'}`
  const hashLinkClass = `${linkBase} text-[#adb5bd] hover:text-[#f8f9fa]`

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 h-16 transition-all duration-200 ${
        scrolled
          ? 'bg-[rgba(26,26,26,0.95)] backdrop-blur-xl border-b border-[rgba(73,80,87,0.5)]'
          : 'bg-[rgba(26,26,26,0.9)] backdrop-blur-xl border-b border-transparent'
      }`}
    >
      <div className="mx-auto flex h-full max-w-[1200px] items-center justify-between px-6">
        {/* Logo */}
        <a href="/" className="flex items-center gap-3" onClick={handleLogoClick}>
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="14" cy="14" r="13" stroke="#e8a645" strokeWidth="1.5" fill="rgba(232,166,69,0.1)"/>
            <ellipse cx="14" cy="18" rx="5" ry="4" fill="#e8a645" opacity="0.8"/>
            <circle cx="10" cy="12" r="1.8" fill="#e8a645"/>
            <circle cx="18" cy="12" r="1.8" fill="#e8a645"/>
            <ellipse cx="14" cy="15" rx="2.5" ry="2" fill="#e8a645" opacity="0.6"/>
          </svg>
          <div className="flex flex-col">
            <span className="text-[15px] font-medium text-[#f8f9fa] leading-tight">
              street_dogs_mp
            </span>
            <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[#6c757d]">
              Mapillary Pipeline v3
            </span>
          </div>
        </a>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center gap-8">
          {pageLinks.map((link) => (
            <NavLink
              key={link.href}
              to={link.href}
              className={pageLinkClass}
              onClick={() => setMobileOpen(false)}
            >
              {link.label}
            </NavLink>
          ))}
          {isHome &&
            homeHashLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={(e) => handleHashClick(e, link.href)}
                className={hashLinkClass}
              >
                {link.label}
              </a>
            ))}
          <a
            href="https://github.com/Alyetama/street_dogs_mp"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center w-9 h-9 rounded-md border border-[#495057] bg-transparent hover:bg-[#2c3034] transition-colors duration-200"
            aria-label="View on GitHub"
          >
            <Github size={16} className="text-[#adb5bd]" />
          </a>
        </div>

        {/* Mobile Hamburger */}
        <button
          className="md:hidden flex items-center justify-center w-9 h-9 text-[#adb5bd]"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div className="md:hidden bg-[rgba(26,26,26,0.98)] backdrop-blur-xl border-b border-[rgba(73,80,87,0.5)]">
          <div className="flex flex-col px-6 py-4 gap-4">
            {pageLinks.map((link) => (
              <NavLink
                key={link.href}
                to={link.href}
                className={({ isActive }) =>
                  `${linkBase} ${isActive ? 'text-[#f8f9fa]' : 'text-[#adb5bd] hover:text-[#f8f9fa]'} py-2`
                }
                onClick={() => setMobileOpen(false)}
              >
                {link.label}
              </NavLink>
            ))}
            {isHome &&
              homeHashLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={(e) => handleHashClick(e, link.href)}
                  className={`${hashLinkClass} py-2`}
                >
                  {link.label}
                </a>
              ))}
            <a
              href="https://github.com/Alyetama/street_dogs_mp"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-[14px] text-[#adb5bd] hover:text-[#f8f9fa] transition-colors duration-200 py-2"
            >
              <Github size={16} />
              View on GitHub
            </a>
          </div>
        </div>
      )}
    </nav>
  )
}
