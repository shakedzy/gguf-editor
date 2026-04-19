import { useState, useEffect, useRef, useCallback } from 'react'
import { formatFloat, formatNumber } from '../../lib/format'

interface Props {
  tensorIndex: number
}

interface Stats {
  min: number
  max: number
  mean: number
  stddev: number
  count: number
}

const HISTOGRAM_BINS = 80

export default function TensorStats({ tensorIndex }: Props) {
  const [stats, setStats] = useState<Stats | null>(null)
  const [histogram, setHistogram] = useState<number[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [histLoading, setHistLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    setLoading(true)
    setStats(null)
    setHistogram(null)
    setError(null)

    window.api
      .getTensorStats(tensorIndex)
      .then((result) => {
        if (result) {
          setStats(result)
          // Now build histogram by sampling dequantized values
          buildHistogram(tensorIndex, result)
        } else {
          setError('Could not compute statistics for this tensor type')
        }
        setLoading(false)
      })
      .catch((err) => {
        setError(String(err))
        setLoading(false)
      })
  }, [tensorIndex])

  const buildHistogram = useCallback(
    async (tIdx: number, s: Stats) => {
      setHistLoading(true)
      // Fetch a sample of floats (first 64K values)
      const sampleSize = 65536
      const buf = await window.api.dequantizeTensor(tIdx, 0, sampleSize)
      if (!buf) {
        setHistLoading(false)
        return
      }
      const floats = new Float32Array(buf)
      const bins = new Array(HISTOGRAM_BINS).fill(0)
      const range = s.max - s.min
      if (range === 0) {
        bins[Math.floor(HISTOGRAM_BINS / 2)] = floats.length
        setHistogram(bins)
        setHistLoading(false)
        return
      }

      for (let i = 0; i < floats.length; i++) {
        const v = floats[i]
        if (!isFinite(v)) continue
        let bin = Math.floor(((v - s.min) / range) * HISTOGRAM_BINS)
        if (bin >= HISTOGRAM_BINS) bin = HISTOGRAM_BINS - 1
        if (bin < 0) bin = 0
        bins[bin]++
      }
      setHistogram(bins)
      setHistLoading(false)
    },
    []
  )

  // Draw histogram on canvas
  useEffect(() => {
    if (!histogram || !canvasRef.current || !stats) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const w = rect.width
    const h = rect.height

    ctx.clearRect(0, 0, w, h)

    const maxCount = Math.max(...histogram)
    if (maxCount === 0) return

    const barWidth = w / HISTOGRAM_BINS
    const padding = 24 // bottom padding for labels

    // Draw bars
    for (let i = 0; i < HISTOGRAM_BINS; i++) {
      const count = histogram[i]
      const barHeight = (count / maxCount) * (h - padding)
      const x = i * barWidth
      const y = h - padding - barHeight

      // Color gradient: blue for normal, brighter for taller bars
      const intensity = count / maxCount
      const r = Math.round(59 + intensity * 40)
      const g = Math.round(130 + intensity * 50)
      const b = Math.round(246)
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
      ctx.fillRect(x + 0.5, y, barWidth - 1, barHeight)
    }

    // Draw mean line
    const range = stats.max - stats.min
    if (range > 0) {
      const meanX = ((stats.mean - stats.min) / range) * w
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      ctx.moveTo(meanX, 0)
      ctx.lineTo(meanX, h - padding)
      ctx.stroke()
      ctx.setLineDash([])

      // +/- stddev shaded region
      const leftX = Math.max(0, ((stats.mean - stats.stddev - stats.min) / range) * w)
      const rightX = Math.min(w, ((stats.mean + stats.stddev - stats.min) / range) * w)
      ctx.fillStyle = 'rgba(245, 158, 11, 0.08)'
      ctx.fillRect(leftX, 0, rightX - leftX, h - padding)
    }

    // X-axis labels
    ctx.fillStyle = '#6b7280'
    ctx.font = '10px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(formatFloat(stats.min, 3), 2, h - 6)
    ctx.textAlign = 'right'
    ctx.fillText(formatFloat(stats.max, 3), w - 2, h - 6)
    ctx.textAlign = 'center'
    ctx.fillStyle = '#f59e0b'
    if (range > 0) {
      const meanX = ((stats.mean - stats.min) / range) * w
      ctx.fillText(`mean: ${formatFloat(stats.mean, 3)}`, Math.max(60, Math.min(w - 60, meanX)), h - 6)
    }
  }, [histogram, stats])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-gray-600 border-t-blue-500 rounded-full mx-auto mb-3" />
          <p className="text-sm">Computing statistics...</p>
          <p className="text-xs text-gray-600 mt-1">This may take a while for large tensors</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-400 text-sm">{error}</div>
    )
  }

  if (!stats) return null

  const statCards = [
    { label: 'Minimum', value: formatFloat(stats.min, 8) },
    { label: 'Maximum', value: formatFloat(stats.max, 8) },
    { label: 'Mean', value: formatFloat(stats.mean, 8) },
    { label: 'Std Dev', value: formatFloat(stats.stddev, 8) },
    { label: 'Range', value: formatFloat(stats.max - stats.min, 8) },
    { label: 'Element Count', value: formatNumber(stats.count) }
  ]

  return (
    <div className="p-6 overflow-auto">
      {/* Stat cards */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {statCards.map(({ label, value }) => (
          <div key={label} className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">{label}</div>
            <div className="text-lg font-mono text-gray-200">{value}</div>
          </div>
        ))}
      </div>

      {/* Histogram */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-gray-300">Value Distribution</h4>
          {histLoading && (
            <span className="text-xs text-gray-500">Computing histogram...</span>
          )}
          {histogram && !histLoading && (
            <span className="text-xs text-gray-600">
              {HISTOGRAM_BINS} bins | sampled {Math.min(stats.count, 65536).toLocaleString()} values
              {stats.count > 65536 && <span className="text-gray-500"> of {formatNumber(stats.count)}</span>}
            </span>
          )}
        </div>
        <div className="relative">
          <canvas
            ref={canvasRef}
            className="w-full h-[200px] rounded"
            style={{ imageRendering: 'auto' }}
          />
          {histLoading && !histogram && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="animate-spin w-6 h-6 border-2 border-gray-600 border-t-blue-500 rounded-full" />
            </div>
          )}
        </div>
        {histogram && (
          <div className="flex items-center gap-4 mt-2 text-[10px] text-gray-500">
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 rounded-sm bg-blue-500" /> count
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-0 border-t-[1.5px] border-dashed border-amber-500" /> mean
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 rounded-sm bg-amber-500/10" /> +/- 1 stddev
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
