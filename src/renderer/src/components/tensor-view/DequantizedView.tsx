import { useState, useEffect, useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { GgufTensorInfo } from '../../store/file-store'
import { formatFloat } from '../../lib/format'

interface Props {
  tensorIndex: number
  tensor: GgufTensorInfo
}

const VALUES_PER_ROW = 8
const CHUNK_SIZE = 8192 // Request 8K floats at a time

export default function DequantizedView({ tensorIndex, tensor }: Props) {
  const [floats, setFloats] = useState<Float32Array | null>(null)
  const [loading, setLoading] = useState(true)
  const parentRef = useRef<HTMLDivElement>(null)

  const totalElements = tensor.dims.reduce((a, b) => a * b, 1)
  const rowCount = Math.ceil(totalElements / VALUES_PER_ROW)

  useEffect(() => {
    setLoading(true)
    setFloats(null)

    const count = Math.min(totalElements, CHUNK_SIZE)
    window.api.dequantizeTensor(tensorIndex, 0, count).then((buf) => {
      if (buf) {
        setFloats(new Float32Array(buf))
      }
      setLoading(false)
    })
  }, [tensorIndex])

  const virtualizer = useVirtualizer({
    count: Math.min(rowCount, floats ? Math.ceil(floats.length / VALUES_PER_ROW) : 0),
    getScrollElement: () => parentRef.current,
    estimateSize: () => 28,
    overscan: 20
  })

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-2 border-gray-600 border-t-blue-500 rounded-full mx-auto mb-3" />
          <p className="text-sm">Dequantizing tensor...</p>
        </div>
      </div>
    )
  }

  if (!floats) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        Failed to dequantize tensor
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 py-2 text-xs text-gray-500 border-b border-gray-800 shrink-0">
        Showing {floats.length.toLocaleString()} / {totalElements.toLocaleString()} values
        {floats.length < totalElements && (
          <span className="ml-2 text-amber-500">(first chunk only)</span>
        )}
      </div>

      <div ref={parentRef} className="flex-1 overflow-auto">
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            width: '100%',
            position: 'relative'
          }}
        >
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const startIdx = virtualRow.index * VALUES_PER_ROW

            return (
              <div
                key={virtualRow.index}
                className="hex-cell flex items-center absolute top-0 left-0 w-full px-4 hover:bg-gray-800/30"
                style={{
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`
                }}
              >
                <span className="w-20 text-gray-600 text-xs select-none shrink-0">
                  [{startIdx}]
                </span>
                <div className="flex-1 flex gap-1">
                  {Array.from({ length: VALUES_PER_ROW }, (_, i) => {
                    const idx = startIdx + i
                    if (idx >= floats.length) return null
                    const val = floats[idx]
                    const isZero = val === 0
                    const isLarge = Math.abs(val) > 10
                    return (
                      <span
                        key={i}
                        className={`w-[100px] text-xs font-mono text-right
                          ${isZero ? 'text-gray-600' : isLarge ? 'text-amber-400' : 'text-gray-300'}`}
                      >
                        {formatFloat(val, 5)}
                      </span>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
