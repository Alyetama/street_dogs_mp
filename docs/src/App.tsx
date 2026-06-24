import { Routes, Route } from 'react-router'
import Home from './pages/Home'
import CLIReference from './pages/CLIReference'
import CoverageAudit from './pages/CoverageAudit'
import Backfill from './pages/Backfill'
import HelperScripts from './pages/HelperScripts'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/cli-reference" element={<CLIReference />} />
      <Route path="/coverage-audit" element={<CoverageAudit />} />
      <Route path="/backfill" element={<Backfill />} />
      <Route path="/helper-scripts" element={<HelperScripts />} />
    </Routes>
  )
}
