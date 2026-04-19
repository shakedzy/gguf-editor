import { useRef, useCallback, useState, useEffect, useMemo } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { getHexExplanation, getStructuredFormula, FormulaPart } from '../../lib/quant-info'
import { GGML_TYPE_NAME } from '../../lib/tensor-types'
import { BLOCK_DEFS, BlockDef } from '../../lib/block-layout'
import { useFileStore } from '../../store/file-store'

interface Props {
  tensorIndex: number
  sizeBytes: number
  tensorType: number
  pendingEdits: Map<number, number>
  onPendingEditsChange: (edits: Map<number, number>) => void
  undoStack: { offset: number; oldVal: number | undefined }[]
  onUndoStackChange: (stack: { offset: number; oldVal: number | undefined }[]) => void
}

const PAGE_SIZE = 4096
// Layout constants (px): offset label + byte cell + separator overhead + right padding
const OFFSET_WIDTH = 72
const BYTE_CELL_WIDTH = 21
const MID_DIV_WIDTH = 9
const SEP_WIDTH = 3
const PADDING = 24

const ROLE_COLOR: Record<string, string> = {
  scale: 'text-cyan-400',
  min: 'text-pink-400',
  sum: 'text-orange-400',
  subscale: 'text-violet-400',
  submin: 'text-amber-400',
  submixed: 'text-green-400',
  highbits: 'text-teal-400',
  signs: 'text-rose-400',
  quants: 'text-gray-300',
  value: 'text-emerald-400'
}

const LEGEND_LABELS: Record<string, string> = {
  scale: 'Scale (d)',
  min: 'Min (dmin)',
  sum: 'Sum',
  subscale: 'Sub-block scales (8 × 6-bit, packed)',
  submin: 'Sub-block mins (8 × 6-bit, packed)',
  submixed: 'Sub-blocks 4–7 scale+min low bits',
  highbits: 'High bits',
  signs: 'Sign bits',
  quants: 'Quantized values',
  value: 'Float value'
}

// ── Helpers ──────────────────────────────────────────────────────────

function getRoleForByte(absoluteOffset: number, def: BlockDef): string {
  const posInBlock = absoluteOffset % def.blockBytes
  for (const seg of def.segments) {
    if (posInBlock >= seg.start && posInBlock < seg.start + seg.len) return seg.role
  }
  return 'quants'
}

function isValueBoundary(absoluteOffset: number, def: BlockDef): boolean {
  if (absoluteOffset === 0) return false
  // For simple types (F32, F16): every blockBytes is one value
  if (def.weightsPerBlock === 1) return absoluteOffset % def.blockBytes === 0
  // For quantized types: no separators — colors are enough
  return false
}

function isBlockBoundary(absoluteOffset: number, def: BlockDef): boolean {
  return absoluteOffset > 0 && absoluteOffset % def.blockBytes === 0
}

function isSubBlockBoundary(absoluteOffset: number, def: BlockDef): boolean {
  if (!def.subBlockBoundaries || absoluteOffset === 0) return false
  const posInBlock = absoluteOffset % def.blockBytes
  return def.subBlockBoundaries.includes(posInBlock)
}

// ── Component ────────────────────────────────────────────────────────

export default function HexEditor({ tensorIndex, sizeBytes, tensorType, pendingEdits, onPendingEditsChange, undoStack, onUndoStackChange }: Props) {
  const saveVersion = useFileStore((s) => s.saveVersion)
  const containerRef = useRef<HTMLDivElement>(null)
  const parentRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(800)

  // Measure container and recalculate on resize
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width)
      }
    })
    ro.observe(el)
    setContainerWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  // Compute bytes per row: multiples of 16, minimum 16
  const bytesPerRow = useMemo(() => {
    const available = containerWidth - OFFSET_WIDTH - PADDING
    // Each byte takes BYTE_CELL_WIDTH, plus one MID_DIV at the halfway point per 16-byte group
    // For N bytes: N * BYTE_CELL_WIDTH + floor(N/16) * MID_DIV_WIDTH + separators
    // Approximate: each 16-byte group takes 16 * 21 + 9 = 345px
    const perGroup = 16 * BYTE_CELL_WIDTH + MID_DIV_WIDTH
    const groups = Math.max(1, Math.floor(available / perGroup))
    return groups * 16
  }, [containerWidth])

  const rowCount = Math.ceil(sizeBytes / bytesPerRow)
  const [cache, setCache] = useState<Map<number, Uint8Array>>(new Map())
  const [editingOffset, setEditingOffset] = useState<number | null>(null)
  const [editHex, setEditHex] = useState('')
  const [showExplanation, setShowExplanation] = useState(true)

  const blockDef = BLOCK_DEFS[tensorType]
  // If only one unique role, don't color — use white
  const uniqueRoles = blockDef ? [...new Set(blockDef.segments.map((s) => s.role))] : []
  const singleRole = uniqueRoles.length <= 1

  useEffect(() => {
    setCache(new Map())
    setEditingOffset(null)
  }, [tensorIndex])

  // After save: flush byte cache so it re-reads from the saved file
  useEffect(() => {
    if (saveVersion === 0) return
    setCache(new Map())
  }, [saveVersion])

  // Cmd+Z / Ctrl+Z undo handler
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
        e.preventDefault()
        if (undoStack.length === 0) return
        const last = undoStack[undoStack.length - 1]
        const next = new Map(pendingEdits)
        if (last.oldVal === undefined) {
          next.delete(last.offset)
        } else {
          next.set(last.offset, last.oldVal)
        }
        onPendingEditsChange(next)
        onUndoStackChange(undoStack.slice(0, -1))
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [undoStack, pendingEdits, onUndoStackChange, onPendingEditsChange])

  const fetchPage = useCallback(
    async (pageStart: number) => {
      if (cache.has(pageStart)) return
      const result = await window.api.readTensorChunk(tensorIndex, pageStart, PAGE_SIZE)
      if (result) {
        setCache((prev) => {
          const next = new Map(prev)
          next.set(pageStart, new Uint8Array(result))
          return next
        })
      }
    },
    [tensorIndex, cache]
  )

  const getOriginalByteAt = useCallback(
    (offset: number): number | null => {
      const pageStart = Math.floor(offset / PAGE_SIZE) * PAGE_SIZE
      const page = cache.get(pageStart)
      if (!page) {
        fetchPage(pageStart)
        return null
      }
      const idx = offset - pageStart
      return idx < page.length ? page[idx] : null
    },
    [cache, fetchPage]
  )

  const getByteAt = useCallback(
    (offset: number): number | null => {
      if (pendingEdits.has(offset)) return pendingEdits.get(offset)!
      return getOriginalByteAt(offset)
    },
    [getOriginalByteAt, pendingEdits]
  )

  const handleByteClick = (offset: number) => {
    const byte = getByteAt(offset)
    if (byte === null) return
    setEditingOffset(offset)
    setEditHex(byte.toString(16).padStart(2, '0').toUpperCase())
  }

  const handleEditCommit = () => {
    if (editingOffset === null) return
    const val = parseInt(editHex, 16)
    if (isNaN(val) || val < 0 || val > 255) {
      setEditingOffset(null)
      return
    }
    const orig = getOriginalByteAt(editingOffset)
    if (orig !== null && val !== orig) {
      const prevVal = pendingEdits.get(editingOffset)
      onUndoStackChange([...undoStack, { offset: editingOffset, oldVal: prevVal }])
      const next = new Map(pendingEdits)
      next.set(editingOffset, val)
      onPendingEditsChange(next)
    } else if (orig !== null && val === orig) {
      const prevVal = pendingEdits.get(editingOffset)
      if (prevVal !== undefined) {
        onUndoStackChange([...undoStack, { offset: editingOffset, oldVal: prevVal }])
      }
      const next = new Map(pendingEdits)
      next.delete(editingOffset)
      onPendingEditsChange(next)
    }
    if (editingOffset + 1 < sizeBytes) {
      const nextOff = editingOffset + 1
      const nextByte = getByteAt(nextOff)
      setEditingOffset(nextOff)
      setEditHex(nextByte !== null ? nextByte.toString(16).padStart(2, '0').toUpperCase() : '00')
    } else {
      setEditingOffset(null)
    }
  }

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 24,
    overscan: 30
  })

  const explanation = getHexExplanation(tensorType)
  const formula = getStructuredFormula(tensorType)
  const typeName = GGML_TYPE_NAME[tensorType] ?? `type ${tensorType}`

  // Sorted edit offsets for navigation
  const editOffsets = useMemo(
    () => [...pendingEdits.keys()].sort((a, b) => a - b),
    [pendingEdits]
  )

  const scrollToOffset = useCallback((offset: number) => {
    const row = Math.floor(offset / bytesPerRow)
    virtualizer.scrollToIndex(row, { align: 'center' })
  }, [bytesPerRow, virtualizer])

  const [navIndex, setNavIndex] = useState(0)

  const goToEdit = useCallback((dir: 1 | -1) => {
    if (editOffsets.length === 0) return
    const next = (navIndex + dir + editOffsets.length) % editOffsets.length
    setNavIndex(next)
    scrollToOffset(editOffsets[next])
  }, [editOffsets, navIndex, scrollToOffset])

  return (
    <div ref={containerRef} className="h-full flex flex-col">
      {/* Explanation banner */}
      {showExplanation && (
        <div className="px-3 py-2 bg-blue-950/40 border-b border-blue-900/50 text-xs text-blue-300/80 leading-relaxed shrink-0">
          <div className="flex items-start justify-between gap-2">
            <p>{explanation}</p>
            <button onClick={() => setShowExplanation(false)} className="text-blue-500/50 hover:text-blue-400 shrink-0 mt-0.5">x</button>
          </div>
        </div>
      )}

      {/* Formula card */}
      {showExplanation && formula && (
        <div className="px-3 py-2 border-b border-gray-800 shrink-0 bg-gray-900/50">
          <div className="flex items-baseline gap-3">
            <span className="text-[10px] text-gray-500 uppercase tracking-wider shrink-0">{typeName}</span>
            <span className="font-mono text-sm">
              {formula.equation.map((part, i) => (
                <FormulaSpan key={i} part={part} />
              ))}
            </span>
          </div>
          <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5">
            {formula.variables.map((v, i) => (
              <span key={i} className="text-[10px] text-gray-500">
                <span className="font-mono">
                  {v.parts.map((p, j) => (
                    <FormulaSpan key={j} part={p} />
                  ))}
                </span>
                {' = '}{v.desc}
              </span>
            ))}
          </div>
          {formula.note && (
            <div className="mt-1 text-[10px] text-gray-600 italic">{formula.note}</div>
          )}
        </div>
      )}

      {/* Status bar */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-800 text-xs text-gray-500 shrink-0">
        <span>
          {sizeBytes.toLocaleString()} bytes
          {pendingEdits.size > 0 && (
            <span className="ml-2 text-blue-400">
              {pendingEdits.size} modified
              {undoStack.length > 0 && <span className="text-gray-500 ml-1">| Cmd+Z to undo</span>}
            </span>
          )}
        </span>
        <div className="flex items-center gap-3">
          {pendingEdits.size > 0 && (
            <div className="flex items-center gap-1">
              <button onClick={() => goToEdit(-1)} className="text-blue-400 hover:text-blue-300 px-1">&lt;</button>
              <span className="text-xs text-gray-500">{navIndex + 1}/{editOffsets.length}</span>
              <button onClick={() => goToEdit(1)} className="text-blue-400 hover:text-blue-300 px-1">&gt;</button>
            </div>
          )}
          {!showExplanation && (
            <button onClick={() => setShowExplanation(true)} className="text-gray-600 hover:text-gray-400">Show info</button>
          )}
          <span className="text-gray-600">Click any byte to edit</span>
        </div>
      </div>

      {/* Color legend — only show when multiple roles */}
      {!singleRole && uniqueRoles.length > 0 && (
        <div className="flex items-center gap-4 px-3 py-1 border-b border-gray-800 text-[10px] shrink-0 flex-wrap">
          {uniqueRoles.map((role) => (
            <span key={role} className="flex items-center gap-1.5">
              <span className={`font-mono font-bold ${ROLE_COLOR[role]}`}>AB</span>
              <span className="text-gray-500">{LEGEND_LABELS[role] ?? role}</span>
            </span>
          ))}
          {blockDef && blockDef.weightsPerBlock > 1 && (
            <span className="flex items-center gap-1.5 ml-1">
              <span className="w-px h-4 bg-gray-400" />
              <span className="text-gray-500">= block ({blockDef.blockBytes}B = {blockDef.weightsPerBlock} weights)</span>
            </span>
          )}
        </div>
      )}

      {/* Column headers */}
      <div className="hex-cell flex items-center py-1 border-b border-gray-800 text-gray-600 select-none shrink-0">
        <span className="w-[72px] pl-2 text-xs shrink-0">Offset</span>
        <div className="flex-1 flex items-center">
          {Array.from({ length: bytesPerRow }, (_, i) => (
            <span key={i} className="flex items-center">
              {i > 0 && i % 16 === 0 && <span className="w-[9px] text-center text-gray-800 shrink-0">|</span>}
              <span className="w-[21px] text-center text-xs shrink-0">
                {(i % 16).toString(16).toUpperCase().padStart(2, '0')}
              </span>
            </span>
          ))}
        </div>
      </div>

      {/* Hex rows */}
      <div ref={parentRef} className="flex-1 overflow-auto">
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            width: '100%',
            position: 'relative'
          }}
        >
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const rowOffset = virtualRow.index * bytesPerRow
            const bytesInRow = Math.min(bytesPerRow, sizeBytes - rowOffset)

            return (
              <div
                key={virtualRow.index}
                className="hex-cell flex items-center absolute top-0 left-0 w-full hover:bg-gray-800/20"
                style={{
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`
                }}
              >
                {/* Offset */}
                <span className="w-[72px] text-gray-600 select-none shrink-0 pl-2">
                  {rowOffset.toString(16).padStart(8, '0')}
                </span>

                {/* Hex bytes — fixed-width cells for alignment */}
                <div className="flex-1 flex items-center">
                  {Array.from({ length: bytesPerRow }, (_, i) => {
                    // Group divider between each 16-byte group
                    const isGroupDiv = i > 0 && i % 16 === 0

                    if (i >= bytesInRow) {
                      return (
                        <span key={i} className="flex items-center">
                          {isGroupDiv && <span className="w-[9px] text-center text-gray-800 shrink-0">|</span>}
                          <span className="w-[21px] shrink-0" />
                        </span>
                      )
                    }

                    const off = rowOffset + i
                    const byte = getByteAt(off)
                    const edited = pendingEdits.has(off)

                    // Color: white if single role, otherwise role-based
                    const role = blockDef ? getRoleForByte(off, blockDef) : null
                    const color = singleRole ? 'text-gray-300' : role ? ROLE_COLOR[role] : 'text-gray-300'

                    // Three separator types: block boundaries, value boundaries, and
                    // sub-block boundaries (between logical sub-blocks inside a k-quant).
                    const showBlockSep = blockDef && isBlockBoundary(off, blockDef)
                    const showValSep = blockDef && !showBlockSep && isValueBoundary(off, blockDef)
                    const showSubBlockSep = blockDef && !showBlockSep && !showValSep && isSubBlockBoundary(off, blockDef)
                    const anySep = showBlockSep || showValSep || showSubBlockSep

                    // For value separators: also show at start of row if the
                    // previous row's last byte ended mid-value
                    const showValSepAtRowStart = !anySep && i === 0 && blockDef && blockDef.weightsPerBlock === 1 &&
                      rowOffset > 0 && (rowOffset % blockDef.blockBytes) === 0

                    return (
                      <span key={i} className="flex items-center">
                        {/* Midpoint | divider — only when no structural separator */}
                        {isGroupDiv && !anySep && (
                          <span className="w-[9px] text-center text-gray-800 shrink-0">|</span>
                        )}
                        {/* Block boundary: bright */}
                        {isGroupDiv && showBlockSep && (
                          <span className="w-[9px] flex justify-center shrink-0">
                            <span className="w-px h-4 bg-gray-400 shrink-0" />
                          </span>
                        )}
                        {!isGroupDiv && showBlockSep && (
                          <span className="w-[3px] flex justify-center shrink-0">
                            <span className="w-px h-4 bg-gray-400 shrink-0" />
                          </span>
                        )}
                        {/* Value boundary (F32/F16 per-float) */}
                        {isGroupDiv && showValSep && (
                          <span className="w-[9px] flex justify-center shrink-0">
                            <span className="w-px h-3 bg-gray-600 shrink-0" />
                          </span>
                        )}
                        {!isGroupDiv && (showValSep || showValSepAtRowStart) && (
                          <span className="w-[3px] flex justify-center shrink-0">
                            <span className="w-px h-3 bg-gray-600 shrink-0" />
                          </span>
                        )}
                        {/* Sub-block boundary at a group-div slot: thin line inside the pre-existing 9px gap */}
                        {isGroupDiv && showSubBlockSep && (
                          <span className="w-[9px] flex justify-center shrink-0">
                            <span className="w-px h-3 bg-gray-700 shrink-0" />
                          </span>
                        )}
                        {/* Sub-block boundary inside a row: draw as a left border on the byte cell so spacing stays aligned with the header. */}
                        {editingOffset === off ? (
                          <input
                            type="text"
                            value={editHex}
                            onChange={(e) => setEditHex(e.target.value.toUpperCase().replace(/[^0-9A-F]/g, '').slice(0, 2))}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); handleEditCommit() }
                              if (e.key === 'Escape') setEditingOffset(null)
                            }}
                            onBlur={handleEditCommit}
                            autoFocus
                            className="w-[21px] bg-blue-900/50 border border-blue-500 rounded-sm text-center focus:outline-none px-0 text-blue-300 shrink-0"
                            maxLength={2}
                          />
                        ) : (
                          <span
                            className={`w-[21px] text-center rounded-sm cursor-pointer hover:bg-white/10 shrink-0
                              ${!isGroupDiv && showSubBlockSep ? 'border-l border-gray-700' : ''}
                              ${edited ? `${color} bg-red-700/30 ring-1 ring-red-500/40` : byte !== null ? color : 'text-gray-700'}`}
                            onClick={() => byte !== null && handleByteClick(off)}
                          >
                            {byte !== null ? byte.toString(16).padStart(2, '0').toUpperCase() : '--'}
                          </span>
                        )}
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

function FormulaSpan({ part }: { part: FormulaPart }) {
  const color = part.role ? ROLE_COLOR[part.role] ?? 'text-gray-200' : 'text-gray-200'
  return (
    <span className={color}>
      {part.text}
      {part.sub && <sub className="text-[0.65em]">{part.sub}</sub>}
      {part.sup && <sup className="text-[0.65em]">{part.sup}</sup>}
    </span>
  )
}
