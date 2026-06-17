import { useRef, useEffect } from 'react'

interface XYZ {
  x: number
  y: number
  z: number
}

interface City {
  lat: number
  lng: number
  xyz: XYZ
}

interface SurfaceDot {
  lat: number
  lng: number
  xyz: XYZ
  brightness: number
}

interface Arc {
  path: XYZ[]
  progress: number
  speed: number
  tail: XYZ[]
}

function latLngToXYZ(lat: number, lng: number, radius: number): XYZ {
  const phi = (90 - lat) * (Math.PI / 180)
  const theta = (lng + 180) * (Math.PI / 180)
  return {
    x: -(radius * Math.sin(phi) * Math.cos(theta)),
    y: radius * Math.cos(phi),
    z: radius * Math.sin(phi) * Math.sin(theta),
  }
}

function rotateY(point: XYZ, angle: number): XYZ {
  const cos = Math.cos(angle)
  const sin = Math.sin(angle)
  return {
    x: point.x * cos - point.z * sin,
    y: point.y,
    z: point.x * sin + point.z * cos,
  }
}

function orthographicProject(point: XYZ, centerX: number, centerY: number): { x: number; y: number } {
  return { x: centerX + point.x, y: centerY - point.y }
}

function generateCity(radius: number): City {
  const lat = (Math.random() - 0.5) * 160
  const lng = (Math.random() - 0.5) * 360
  return { lat, lng, xyz: latLngToXYZ(lat, lng, radius) }
}

function getIntermediatePoints(start: XYZ, end: XYZ, numPoints: number): XYZ[] {
  const dotProduct = (start.x * end.x + start.y * end.y + start.z * end.z) /
    (Math.sqrt(start.x * start.x + start.y * start.y + start.z * start.z) *
     Math.sqrt(end.x * end.x + end.y * end.y + end.z * end.z))
  const angle = Math.acos(Math.max(-1, Math.min(1, dotProduct)))

  if (angle < 0.01) return [start]

  const points: XYZ[] = []
  for (let i = 0; i <= numPoints; i++) {
    const t = i / numPoints
    const sinAngle = Math.sin(angle)
    const a = Math.sin((1 - t) * angle) / sinAngle
    const b = Math.sin(t * angle) / sinAngle
    points.push({
      x: start.x * a + end.x * b,
      y: start.y * a + end.y * b,
      z: start.z * a + end.z * b,
    })
  }
  return points
}

export default function GlobeCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)

    let GLOBE_RADIUS = Math.min(window.innerWidth, window.innerHeight) * 0.38
    let PROJECTION_CENTER = { x: window.innerWidth * 0.65, y: window.innerHeight * 0.5 }

    const ROTATION_SPEED = 0.0008
    const CITY_COUNT = 40
    const ARC_COUNT = 25
    const DOT_DENSITY = 400
    const TAIL_LENGTH = 12

    let globeRotation = 0
    const cities: City[] = []
    const arcs: Arc[] = []
    const surfaceDots: SurfaceDot[] = []

    // Generate cities
    for (let i = 0; i < CITY_COUNT; i++) {
      cities.push(generateCity(GLOBE_RADIUS))
    }

    // Generate surface dots
    for (let i = 0; i < DOT_DENSITY; i++) {
      const lat = (Math.random() - 0.5) * 170
      const lng = (Math.random() - 0.5) * 360
      const r = GLOBE_RADIUS * (0.98 + Math.random() * 0.04)
      surfaceDots.push({
        lat,
        lng,
        xyz: latLngToXYZ(lat, lng, r),
        brightness: 0.1 + Math.random() * 0.4,
      })
    }

    // Initialize arcs
    for (let i = 0; i < ARC_COUNT; i++) {
      const startCity = cities[Math.floor(Math.random() * cities.length)]
      let endCity = cities[Math.floor(Math.random() * cities.length)]
      while (endCity === startCity) {
        endCity = cities[Math.floor(Math.random() * cities.length)]
      }
      const path = getIntermediatePoints(startCity.xyz, endCity.xyz, 80)
      arcs.push({
        path,
        progress: Math.random(),
        speed: 0.003 + Math.random() * 0.006,
        tail: [],
      })
    }

    function resize() {
      if (!canvas || !ctx) return
      GLOBE_RADIUS = Math.min(window.innerWidth, window.innerHeight) * 0.38
      PROJECTION_CENTER = { x: window.innerWidth * 0.65, y: window.innerHeight * 0.5 }
      canvas.width = window.innerWidth * dpr
      canvas.height = window.innerHeight * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    resize()
    window.addEventListener('resize', resize)

    function draw() {
      if (!ctx) return
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight)

      globeRotation += ROTATION_SPEED

      // Draw globe base circle
      ctx.beginPath()
      ctx.arc(PROJECTION_CENTER.x, PROJECTION_CENTER.y, GLOBE_RADIUS, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(42, 42, 42, 0.8)'
      ctx.fill()
      ctx.strokeStyle = 'rgba(73, 80, 87, 0.5)'
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw surface dots
      for (const dot of surfaceDots) {
        const rotated = rotateY(dot.xyz, globeRotation)
        if (rotated.z < -10) continue
        const screen = orthographicProject(rotated, PROJECTION_CENTER.x, PROJECTION_CENTER.y)
        const alpha = dot.brightness * Math.max(0.1, (rotated.z + GLOBE_RADIUS) / (2 * GLOBE_RADIUS))
        ctx.fillStyle = `rgba(232, 166, 69, ${alpha * 0.6})`
        ctx.fillRect(screen.x - 1, screen.y - 1, 2, 2)
      }

      // Draw arc paths and traveling particles
      for (const arc of arcs) {
        // Draw path line
        ctx.beginPath()
        let firstVisible = true
        for (let i = 0; i < arc.path.length; i++) {
          const pt = rotateY(arc.path[i], globeRotation)
          if (pt.z < -GLOBE_RADIUS * 0.7) {
            firstVisible = true
            continue
          }
          const screen = orthographicProject(pt, PROJECTION_CENTER.x, PROJECTION_CENTER.y)
          if (firstVisible) {
            ctx.moveTo(screen.x, screen.y)
            firstVisible = false
          } else {
            ctx.lineTo(screen.x, screen.y)
          }
        }
        ctx.strokeStyle = 'rgba(232, 166, 69, 0.08)'
        ctx.lineWidth = 1
        ctx.stroke()

        // Update progress
        arc.progress += arc.speed
        if (arc.progress >= 1) {
          const startCity = cities[Math.floor(Math.random() * cities.length)]
          let endCity = cities[Math.floor(Math.random() * cities.length)]
          while (endCity === startCity) {
            endCity = cities[Math.floor(Math.random() * cities.length)]
          }
          arc.path = getIntermediatePoints(startCity.xyz, endCity.xyz, 80)
          arc.progress = 0
          arc.tail = []
        }

        // Calculate current position
        const posIdx = Math.floor(arc.progress * (arc.path.length - 1))
        const currentPoint = arc.path[posIdx]
        if (!currentPoint) continue

        const rotatedCurrent = rotateY(currentPoint, globeRotation)
        const screenCurrent = orthographicProject(rotatedCurrent, PROJECTION_CENTER.x, PROJECTION_CENTER.y)

        // Add to tail
        arc.tail.push(currentPoint)
        if (arc.tail.length > TAIL_LENGTH) {
          arc.tail.shift()
        }

        // Draw tail
        for (let i = 0; i < arc.tail.length; i++) {
          const tailPt = rotateY(arc.tail[i], globeRotation)
          if (tailPt.z < -GLOBE_RADIUS * 0.7) continue
          const screen = orthographicProject(tailPt, PROJECTION_CENTER.x, PROJECTION_CENTER.y)
          const tailAlpha = (i / arc.tail.length) * 0.4
          const size = 1 + (i / arc.tail.length) * 2.5
          ctx.fillStyle = `rgba(232, 166, 69, ${tailAlpha})`
          ctx.beginPath()
          ctx.arc(screen.x, screen.y, size, 0, Math.PI * 2)
          ctx.fill()
        }

        // Draw head
        if (rotatedCurrent.z >= -GLOBE_RADIUS * 0.7) {
          ctx.fillStyle = '#e8a645'
          ctx.beginPath()
          ctx.arc(screenCurrent.x, screenCurrent.y, 2.5, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      // Draw city markers
      for (const city of cities) {
        const rotated = rotateY(city.xyz, globeRotation)
        if (rotated.z < -50) continue
        const screen = orthographicProject(rotated, PROJECTION_CENTER.x, PROJECTION_CENTER.y)
        const alpha = Math.max(0.15, Math.min(1, (rotated.z + GLOBE_RADIUS) / (GLOBE_RADIUS * 0.8)))

        // Outer glow
        ctx.fillStyle = `rgba(232, 166, 69, ${alpha * 0.2})`
        ctx.beginPath()
        ctx.arc(screen.x, screen.y, 5, 0, Math.PI * 2)
        ctx.fill()

        // Core dot
        ctx.fillStyle = `rgba(248, 249, 250, ${alpha})`
        ctx.beginPath()
        ctx.arc(screen.x, screen.y, 2, 0, Math.PI * 2)
        ctx.fill()
      }

      animRef.current = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      cancelAnimationFrame(animRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-label="Animated 3D globe showing data pipeline routes connecting cities across the planet"
      role="img"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
      }}
    />
  )
}
