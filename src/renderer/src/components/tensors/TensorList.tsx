import { useState, useRef, useMemo } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useFileStore } from '../../store/file-store'
import { formatBytes, formatShape } from '../../lib/format'
import { GGML_TYPE_NAME, getTypeBadgeColor } from '../../lib/tensor-types'

type SortKey = 'name' | 'type' | 'size' | 'shape'
type SortDir = 'asc' | 'desc'

export default function TensorList() {
  const { fileInfo, selectedTensorIndex, selectTensor } = useFileStore()
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('name')
  const [sortDir, setSortDir] = useState<SortDir>('asc')
  const parentRef = useRef<HTMLDivElement>(null)

  const tensors = fileInfo?.tensors ?? []

  const filtered = useMemo(() => {
    let list = tensors.map((t, i) => ({ ...t, originalIndex: i }))

    if (search) {
      const q = search.toLowerCase()
      list = list.filter((t) => t.name.toLowerCase().includes(q))
    }

    list.sort((a, b) => {
      let cmp = 0
      switch (sortKey) {
        case 'name':
          cmp = a.name.localeCompare(b.name)
          break
        case 'type':
          cmp = a.type - b.type
          break
        case 'size':
          cmp = a.sizeBytes - b.sizeBytes
          break
        case 'shape':
          cmp = (a.dims[0] ?? 0) - (b.dims[0] ?? 0)
          break
      }
      return sortDir === 'asc' ? cmp : -cmp
    })

    return list
  }, [tensors, search, sortKey, sortDir])

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 40,
    overscan: 20
  })

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const sortArrow = (key: SortKey) => {
    if (sortKey !== key) return ''
    return sortDir === 'asc' ? ' ^' : ' v'
  }

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="flex items-center gap-3 mb-4">
        <h2 className="text-xl font-bold text-gray-100">Tensors</h2>
        <span className="text-sm text-gray-500">
          {filtered.length} / {tensors.length}
        </span>
      </div>

      <input
        type="text"
        placeholder="Search tensors by name..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full px-3 py-2 mb-3 bg-gray-900 border border-gray-700 rounded-md text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
      />

      {/* Header row */}
      <div className="flex text-xs text-gray-500 uppercase px-3 py-2 border-b border-gray-800 shrink-0">
        <button
          className="flex-1 text-left hover:text-gray-300"
          onClick={() => toggleSort('name')}
        >
          Name{sortArrow('name')}
        </button>
        <button
          className="w-24 text-left hover:text-gray-300"
          onClick={() => toggleSort('shape')}
        >
          Shape{sortArrow('shape')}
        </button>
        <button
          className="w-20 text-left hover:text-gray-300"
          onClick={() => toggleSort('type')}
        >
          Type{sortArrow('type')}
        </button>
        <button
          className="w-24 text-right hover:text-gray-300"
          onClick={() => toggleSort('size')}
        >
          Size{sortArrow('size')}
        </button>
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
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const tensor = filtered[virtualRow.index]
            const isSelected = tensor.originalIndex === selectedTensorIndex

            return (
              <div
                key={tensor.originalIndex}
                className={`flex items-center px-3 py-2 text-sm cursor-pointer border-b border-gray-800/30 transition-colors absolute top-0 left-0 w-full
                  ${isSelected ? 'bg-blue-600/20 text-blue-300' : 'hover:bg-gray-800/40 text-gray-300'}`}
                style={{
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`
                }}
                onClick={() => selectTensor(tensor.originalIndex)}
              >
                <div className="flex-1 font-mono text-xs truncate pr-2">
                  {tensor.name}
                </div>
                <div className="w-24 text-xs text-gray-400 font-mono">
                  {formatShape(tensor.dims)}
                </div>
                <div className="w-20">
                  <span
                    className={`px-1.5 py-0.5 rounded text-xs font-mono ${getTypeBadgeColor(tensor.type)}`}
                  >
                    {GGML_TYPE_NAME[tensor.type] ?? `?${tensor.type}`}
                  </span>
                </div>
                <div className="w-24 text-right text-xs text-gray-400">
                  {formatBytes(tensor.sizeBytes)}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
