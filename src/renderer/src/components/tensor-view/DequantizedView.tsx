import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { GgufTensorInfo } from '../../store/file-store'
import { formatFloat } from '../../lib/format'
import { getFloatExplanation, getQuantExplanation } from '../../lib/quant-info'
import { GGML_TYPE_NAME } from '../../lib/tensor-types'

interface Props {
  tensorIndex: number
  tensor: GgufTensorInfo
  pendingEdits?: Map<number, number>
}

const FLOAT_CELL_WIDTH = 105 // px per float value column
const INDEX_WIDTH = 100 // px for the index label
const FLOAT_PADDING = 32 // px total horizontal padding

// Convert flat index to multi-dimensional index given dims
// GGUF stores in row-major order: last dim varies fastest
function flatToNd(flatIdx: number, dims: number[]): number[] {
  if (dims.length <= 1) return [flatIdx]
  const nd: number[] = new Array(dims.length)
  let remaining = flatIdx
  for (let i = dims.length - 1; i >= 0; i--) {
    nd[i] = remaining % dims[i]
    remaining = Math.floor(remaining / dims[i])
  }
  return nd
}

function formatNdIndex(flatIdx: number, dims: number[]): string {
  if (dims.length <= 1) return `[${flatIdx}]`
  return '[' + flatToNd(flatIdx, dims).join(', ') + ']'
}
const PAGE_FLOATS = 8192 // Floats per page fetched from main process

// We need to know byte size per float to compute byte offsets for pagination.
// For block-quantized types, we fetch by block-aligned byte chunks.
const QUANT_BLOCK: Record<number, { blockBytes: number; weightsPerBlock: number }> = {
  0: { blockBytes: 4, weightsPerBlock: 1 },
  1: { blockBytes: 2, weightsPerBlock: 1 },
  2: { blockBytes: 18, weightsPerBlock: 32 },
  3: { blockBytes: 20, weightsPerBlock: 32 },
  6: { blockBytes: 22, weightsPerBlock: 32 },
  7: { blockBytes: 24, weightsPerBlock: 32 },
  8: { blockBytes: 34, weightsPerBlock: 32 },
  9: { blockBytes: 36, weightsPerBlock: 32 },
  12: { blockBytes: 144, weightsPerBlock: 256 },
  14: { blockBytes: 210, weightsPerBlock: 256 }
}

function floatOffsetToByteOffset(floatIdx: number, type: number): number {
  const info = QUANT_BLOCK[type]
  if (!info) return 0
  const blockIdx = Math.floor(floatIdx / info.weightsPerBlock)
  return blockIdx * info.blockBytes
}

function floatCountToByteLen(floatCount: number, type: number): number {
  const info = QUANT_BLOCK[type]
  if (!info) return 0
  const blocks = Math.ceil(floatCount / info.weightsPerBlock)
  return blocks * info.blockBytes
}

export default function DequantizedView({ tensorIndex, tensor, pendingEdits }: Props) {
  const [pages, setPages] = useState<Map<number, Float32Array>>(new Map())
  const [loadingPages, setLoadingPages] = useState<Set<number>>(new Set())
  const [showExplanation, setShowExplanation] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)
  const parentRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(600)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) setContainerWidth(entry.contentRect.width)
    })
    ro.observe(el)
    setContainerWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  const valuesPerRow = useMemo(() => {
    const available = containerWidth - INDEX_WIDTH - FLOAT_PADDING
    return Math.max(1, Math.floor(available / FLOAT_CELL_WIDTH))
  }, [containerWidth])

  const totalElements = tensor.dims.reduce((a, b) => a * b, 1)
  const totalRows = Math.ceil(totalElements / valuesPerRow)

  // Compute which float indices are affected by pending byte edits
  const affectedFloats = useMemo(() => {
    if (!pendingEdits || pendingEdits.size === 0) return null
    const info = QUANT_BLOCK[tensor.type]
    if (!info) return null
    const affected = new Set<number>()
    for (const byteOff of pendingEdits.keys()) {
      const blockIdx = Math.floor(byteOff / info.blockBytes)
      const floatStart = blockIdx * info.weightsPerBlock
      const floatEnd = Math.min(floatStart + info.weightsPerBlock, totalElements)
      for (let f = floatStart; f < floatEnd; f++) affected.add(f)
    }
    return affected
  }, [pendingEdits, tensor.type, totalElements])

  // Reset on tensor change
  useEffect(() => {
    setPages(new Map())
    setLoadingPages(new Set())
  }, [tensorIndex])

  // Invalidate cached pages when edits change
  const editsVersion = pendingEdits?.size ?? 0
  useEffect(() => {
    setPages(new Map())
    setLoadingPages(new Set())
  }, [editsVersion])

  // Convert pendingEdits map to array format for IPC
  const editsArray = useMemo((): [number, number][] | undefined => {
    if (!pendingEdits || pendingEdits.size === 0) return undefined
    return Array.from(pendingEdits.entries())
  }, [pendingEdits, editsVersion])

  // Load a page of floats starting at the given float index
  const loadPage = useCallback(
    async (pageStart: number) => {
      if (pages.has(pageStart) || loadingPages.has(pageStart)) return
      setLoadingPages((prev) => {
        const next = new Set(prev)
        next.add(pageStart)
        return next
      })

      const byteOffset = floatOffsetToByteOffset(pageStart, tensor.type)
      const byteLen = floatCountToByteLen(PAGE_FLOATS, tensor.type)
      const buf = await window.api.dequantizeTensor(tensorIndex, byteOffset, PAGE_FLOATS, editsArray)

      if (buf) {
        setPages((prev) => {
          const next = new Map(prev)
          next.set(pageStart, new Float32Array(buf))
          return next
        })
      }
      setLoadingPages((prev) => {
        const next = new Set(prev)
        next.delete(pageStart)
        return next
      })
    },
    [tensorIndex, tensor.type, pages, loadingPages, editsArray]
  )

  const getFloat = useCallback(
    (idx: number): number | null => {
      const pageStart = Math.floor(idx / PAGE_FLOATS) * PAGE_FLOATS
      const page = pages.get(pageStart)
      if (!page) {
        loadPage(pageStart)
        return null
      }
      const pageIdx = idx - pageStart
      return pageIdx < page.length ? page[pageIdx] : null
    },
    [pages, loadPage]
  )

  const virtualizer = useVirtualizer({
    count: totalRows,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 28,
    overscan: 20
  })

  const explanation = getFloatExplanation(tensor.type)
  const quantInfo = getQuantExplanation(tensor.type)
  const typeName = GGML_TYPE_NAME[tensor.type] ?? `type ${tensor.type}`

  const loadedCount = Array.from(pages.values()).reduce((sum, p) => sum + p.length, 0)

  return (
    <div ref={containerRef} className="h-full flex flex-col">
      {/* Explanation banner */}
      {showExplanation && (
        <div className="px-3 py-2 bg-emerald-950/40 border-b border-emerald-900/50 text-xs leading-relaxed shrink-0">
          <div className="flex items-start justify-between gap-2">
            <div className="text-emerald-300/80">
              <p>{explanation}</p>
              {quantInfo && tensor.type !== 0 && tensor.type !== 1 && (
                <p className="mt-1 text-emerald-400/60">
                  Block layout: {quantInfo.blockLayout}
                </p>
              )}
            </div>
            <button
              onClick={() => setShowExplanation(false)}
              className="text-emerald-500/50 hover:text-emerald-400 shrink-0 mt-0.5"
            >
              x
            </button>
          </div>
        </div>
      )}

      {/* Status bar */}
      <div className="px-4 py-2 text-xs text-gray-500 border-b border-gray-800 shrink-0 flex items-center justify-between">
        <span>
          {totalElements.toLocaleString()} values total
          <span className="ml-2 text-gray-600">
            ({loadedCount.toLocaleString()} loaded)
          </span>
          <span className="ml-3 text-gray-600">
            Stored as {typeName}
            {quantInfo && ` (${quantInfo.bytesPerFloat})`}
          </span>
        </span>
        <div className="flex items-center gap-3">
          {!showExplanation && (
            <button onClick={() => setShowExplanation(true)} className="text-gray-600 hover:text-gray-400">
              Show info
            </button>
          )}
          <span className="text-gray-600">Scroll to load more</span>
        </div>
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
            const startIdx = virtualRow.index * valuesPerRow

            return (
              <div
                key={virtualRow.index}
                className="hex-cell flex items-center absolute top-0 left-0 w-full px-4 hover:bg-gray-800/30"
                style={{
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`
                }}
              >
                <span className="text-gray-600 text-[10px] select-none shrink-0 font-mono pr-2 min-w-[80px]">
                  {formatNdIndex(startIdx, tensor.dims)}
                </span>
                <div className="flex-1 flex gap-1">
                  {Array.from({ length: valuesPerRow }, (_, i) => {
                    const idx = startIdx + i
                    if (idx >= totalElements) return null
                    const val = getFloat(idx)
                    if (val === null) {
                      return (
                        <span key={i} className="w-[100px] text-xs font-mono text-right text-gray-700">
                          ...
                        </span>
                      )
                    }
                    const isZero = val === 0
                    const isEdited = affectedFloats?.has(idx) ?? false
                    return (
                      <span
                        key={i}
                        className={`w-[100px] text-xs font-mono text-right
                          ${isEdited ? 'text-amber-400' : isZero ? 'text-gray-600' : 'text-gray-300'}`}
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
