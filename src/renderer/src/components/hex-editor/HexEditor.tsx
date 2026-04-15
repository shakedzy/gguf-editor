import { useRef, useCallback, useState, useEffect } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'

interface Props {
  tensorIndex: number
  sizeBytes: number
}

const BYTES_PER_ROW = 16
const PAGE_SIZE = 4096 // Fetch 4KB at a time

export default function HexEditor({ tensorIndex, sizeBytes }: Props) {
  const parentRef = useRef<HTMLDivElement>(null)
  const rowCount = Math.ceil(sizeBytes / BYTES_PER_ROW)
  const [cache, setCache] = useState<Map<number, Uint8Array>>(new Map())

  // Reset cache when tensor changes
  useEffect(() => {
    setCache(new Map())
  }, [tensorIndex])

  const fetchPage = useCallback(
    async (pageStart: number) => {
      const key = pageStart
      if (cache.has(key)) return
      const result = await window.api.readTensorChunk(
        tensorIndex,
        pageStart,
        PAGE_SIZE
      )
      if (result) {
        setCache((prev) => {
          const next = new Map(prev)
          next.set(key, new Uint8Array(result))
          return next
        })
      }
    },
    [tensorIndex, cache]
  )

  const getByteAt = useCallback(
    (offset: number): number | null => {
      const pageStart = Math.floor(offset / PAGE_SIZE) * PAGE_SIZE
      const page = cache.get(pageStart)
      if (!page) {
        fetchPage(pageStart)
        return null
      }
      const pageOffset = offset - pageStart
      if (pageOffset >= page.length) return null
      return page[pageOffset]
    },
    [cache, fetchPage]
  )

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 24,
    overscan: 30
  })

  return (
    <div ref={parentRef} className="h-full overflow-auto p-2">
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative'
        }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const rowOffset = virtualRow.index * BYTES_PER_ROW
          const bytesInRow = Math.min(BYTES_PER_ROW, sizeBytes - rowOffset)

          const hexCells: string[] = []
          const asciiCells: string[] = []

          for (let i = 0; i < BYTES_PER_ROW; i++) {
            if (i < bytesInRow) {
              const byte = getByteAt(rowOffset + i)
              if (byte !== null) {
                hexCells.push(byte.toString(16).padStart(2, '0').toUpperCase())
                asciiCells.push(byte >= 32 && byte < 127 ? String.fromCharCode(byte) : '.')
              } else {
                hexCells.push('--')
                asciiCells.push(' ')
              }
            } else {
              hexCells.push('  ')
              asciiCells.push(' ')
            }
          }

          return (
            <div
              key={virtualRow.index}
              className="hex-cell flex items-center absolute top-0 left-0 w-full hover:bg-gray-800/30"
              style={{
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`
              }}
            >
              {/* Offset */}
              <span className="w-24 text-gray-600 select-none shrink-0 pl-2">
                {rowOffset.toString(16).padStart(8, '0')}
              </span>

              {/* Hex bytes */}
              <span className="flex-1 text-gray-300 tracking-wider">
                {hexCells.slice(0, 8).join(' ')}
                <span className="text-gray-700 mx-1">|</span>
                {hexCells.slice(8).join(' ')}
              </span>

              {/* ASCII */}
              <span className="w-[140px] text-gray-500 select-none shrink-0 text-right pr-2">
                {asciiCells.join('')}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
