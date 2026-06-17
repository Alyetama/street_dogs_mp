import { Routes, Route } from 'react-router'
import Home from './pages/Home'
import CLIReference from './pages/CLIReference'
import HelperScripts from './pages/HelperScripts'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/cli-reference" element={<CLIReference />} />
      <Route path="/helper-scripts" element={<HelperScripts />} />
    </Routes>
  )
}
