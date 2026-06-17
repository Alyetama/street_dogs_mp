import { useEffect } from 'react'
import Navigation from '../sections/Navigation'
import HeroSection from '../sections/HeroSection'
import WorkflowSection from '../sections/WorkflowSection'
import QuickStartSection from '../sections/QuickStartSection'
import FeaturesSection from '../sections/FeaturesSection'
import FooterSection from '../sections/FooterSection'

export default function Home() {
  useEffect(() => {
    // Initialize smooth scrolling with Lenis-like behavior using native scroll
    document.documentElement.style.scrollBehavior = 'smooth'
    return () => {
      document.documentElement.style.scrollBehavior = 'auto'
    }
  }, [])

  return (
    <div className="min-h-screen bg-[#1a1a1a]">
      <Navigation />
      <HeroSection />
      <WorkflowSection />
      <QuickStartSection />
      <FeaturesSection />
      <FooterSection />
    </div>
  )
}
