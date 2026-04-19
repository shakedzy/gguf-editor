import { useState, useRef, useMemo } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { METADATA_TYPE_NAME } from '../../lib/tensor-types'

interface Props {
  metadataKey: string
  arrayType: number
  values: any[]
  onClose: () => void
  onSave?: (values: any[]) => void
}

export default function ArrayViewer({
  metadataKey,
  arrayType,
  values,
  onClose,
  onSave
}: Props) {
  const [search, setSearch] = useState('')
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editValue, setEditValue] = useState('')
  const [localValues, setLocalValues] = useState<any[]>(values)
  const [dirty, setDirty] = useState(false)
  const parentRef = useRef<HTMLDivElement>(null)

  const filtered = useMemo(() => {
    if (!search) return localValues.map((v, i) => ({ value: v, index: i }))
    const q = search.toLowerCase()
    return localValues
      .map((v, i) => ({ value: v, index: i }))
      .filter(({ value }) => String(value).toLowerCase().includes(q))
  }, [localValues, search])

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32,
    overscan: 40
  })

  const handleEditStart = (idx: number, value: any) => {
    setEditingIndex(idx)
    setEditValue(String(value))
  }

  const handleEditSave = (idx: number) => {
    let parsed: any = editValue
    if ([0, 1, 2, 3, 4, 5, 10, 11].includes(arrayType)) {
      parsed = Number(editValue)
      if (isNaN(parsed)) return
    } else if (arrayType === 6 || arrayType === 12) {
      parsed = parseFloat(editValue)
      if (isNaN(parsed)) return
    } else if (arrayType === 7) {
      parsed = editValue === 'true' || editValue === '1'
    }
    const next = [...localValues]
    next[idx] = parsed
    setLocalValues(next)
    setDirty(true)
    setEditingIndex(null)
  }

  const handleSave = () => {
    if (onSave) onSave(localValues)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-[700px] h-[600px] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-800 shrink-0">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-mono text-sm font-bold text-gray-200 truncate">
              {metadataKey}
            </h3>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-300 text-lg px-2"
            >
              x
            </button>
          </div>
          <div className="flex items-center gap-3 text-xs text-gray-400">
            <span className="px-1.5 py-0.5 rounded bg-gray-800 font-mono">
              ARRAY[{METADATA_TYPE_NAME[arrayType] ?? arrayType}]
            </span>
            <span>{localValues.length.toLocaleString()} items</span>
            {dirty && <span className="text-amber-400">Modified</span>}
          </div>
        </div>

        {/* Search */}
        <div className="px-4 py-2 border-b border-gray-800 shrink-0">
          <input
            type="text"
            placeholder="Search values..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Column headers */}
        <div className="flex text-xs text-gray-500 uppercase px-4 py-1.5 border-b border-gray-800 shrink-0">
          <span className="w-20">Index</span>
          <span className="flex-1">Value</span>
          <span className="w-16" />
        </div>

        {/* Virtualized list */}
        <div ref={parentRef} className="flex-1 overflow-auto">
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              width: '100%',
              position: 'relative'
            }}
          >
            {virtualizer.getVirtualItems().map((vRow) => {
              const { value, index } = filtered[vRow.index]
              const isEditing = editingIndex === index

              return (
                <div
                  key={index}
                  className="flex items-center px-4 text-sm border-b border-gray-800/30 hover:bg-gray-800/30 absolute top-0 left-0 w-full group"
                  style={{
                    height: `${vRow.size}px`,
                    transform: `translateY(${vRow.start}px)`
                  }}
                >
                  <span className="w-20 text-xs text-gray-600 font-mono shrink-0">
                    {index}
                  </span>

                  {isEditing ? (
                    <div className="flex-1 flex gap-2">
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleEditSave(index)
                          if (e.key === 'Escape') setEditingIndex(null)
                        }}
                        autoFocus
                        className="flex-1 px-2 py-0.5 bg-gray-900 border border-blue-500 rounded text-xs text-gray-200 focus:outline-none font-mono"
                      />
                      <button
                        onClick={() => handleEditSave(index)}
                        className="px-2 py-0.5 bg-blue-600 rounded text-xs"
                      >
                        OK
                      </button>
                    </div>
                  ) : (
                    <>
                      <span className="flex-1 font-mono text-xs text-gray-300 truncate">
                        {String(value)}
                      </span>
                      <button
                        onClick={() => handleEditStart(index, value)}
                        className="w-16 text-center px-2 py-0.5 text-xs bg-gray-700 hover:bg-gray-600 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        Edit
                      </button>
                    </>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-gray-800 flex justify-end gap-2 shrink-0">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-gray-700 hover:bg-gray-600 rounded-md"
          >
            {dirty ? 'Discard' : 'Close'}
          </button>
          {dirty && onSave && (
            <button
              onClick={handleSave}
              className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded-md"
            >
              Save Changes
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
