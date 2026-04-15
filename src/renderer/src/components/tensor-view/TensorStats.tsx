import { useState, useEffect } from 'react'
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

export default function TensorStats({ tensorIndex }: Props) {
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setStats(null)
    setError(null)

    window.api
      .getTensorStats(tensorIndex)
      .then((result) => {
        if (result) {
          setStats(result)
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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-gray-600 border-t-blue-500 rounded-full mx-auto mb-3" />
          <p className="text-sm">Computing statistics...</p>
          <p className="text-xs text-gray-600 mt-1">
            This may take a while for large tensors
          </p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-400 text-sm">
        {error}
      </div>
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
    <div className="p-6">
      <div className="grid grid-cols-2 gap-4">
        {statCards.map(({ label, value }) => (
          <div
            key={label}
            className="bg-gray-900 rounded-lg p-4 border border-gray-800"
          >
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">
              {label}
            </div>
            <div className="text-lg font-mono text-gray-200">{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
