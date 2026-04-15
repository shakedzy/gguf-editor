import { useState } from 'react'
import { useFileStore } from '../../store/file-store'
import { METADATA_TYPE_NAME } from '../../lib/tensor-types'
import AddMetadataDialog from './AddMetadataDialog'

function formatValue(value: any, valueType: number): string {
  if (valueType === 9 && value && typeof value === 'object' && value.values) {
    const arr = value.values as any[]
    if (arr.length <= 5) return `[${arr.join(', ')}]`
    return `[${arr.slice(0, 3).join(', ')}, ... (${arr.length} items)]`
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

export default function MetadataEditor() {
  const { fileInfo } = useFileStore()
  const [search, setSearch] = useState('')
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [showAdd, setShowAdd] = useState(false)

  if (!fileInfo) return null

  const filtered = fileInfo.metadata.filter((kv) =>
    kv.key.toLowerCase().includes(search.toLowerCase())
  )

  const handleEdit = (key: string, currentValue: any) => {
    setEditingKey(key)
    setEditValue(String(currentValue))
  }

  const handleSaveEdit = async (key: string, valueType: number) => {
    let parsed: any = editValue
    if ([0, 1, 2, 3, 4, 5, 10, 11].includes(valueType)) {
      parsed = Number(editValue)
      if (isNaN(parsed)) return
    } else if (valueType === 6 || valueType === 12) {
      parsed = parseFloat(editValue)
      if (isNaN(parsed)) return
    } else if (valueType === 7) {
      parsed = editValue === 'true' || editValue === '1'
    }

    await window.api.updateMetadata(key, parsed, valueType)
    useFileStore.getState().updateMetadata(key, parsed, valueType)
    setEditingKey(null)
  }

  const handleDelete = async (key: string) => {
    await window.api.deleteMetadata(key)
    useFileStore.getState().deleteMetadata(key)
  }

  return (
    <div className="p-6 space-y-4 h-full flex flex-col">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-bold text-gray-100">Metadata</h2>
        <span className="text-sm text-gray-500">
          {filtered.length} / {fileInfo.metadata.length} keys
        </span>
        <div className="flex-1" />
        <button
          onClick={() => setShowAdd(true)}
          className="px-3 py-1.5 text-sm font-medium bg-green-600 hover:bg-green-500 rounded-md transition-colors"
        >
          + Add
        </button>
      </div>

      <input
        type="text"
        placeholder="Search metadata keys..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-md text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500"
      />

      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-gray-950">
            <tr className="border-b border-gray-800">
              <th className="text-left px-3 py-2 text-xs text-gray-500 uppercase w-[40%]">
                Key
              </th>
              <th className="text-left px-3 py-2 text-xs text-gray-500 uppercase w-20">
                Type
              </th>
              <th className="text-left px-3 py-2 text-xs text-gray-500 uppercase">
                Value
              </th>
              <th className="px-3 py-2 w-20" />
            </tr>
          </thead>
          <tbody>
            {filtered.map((kv) => (
              <tr
                key={kv.key}
                className="border-b border-gray-800/50 hover:bg-gray-800/30 group"
              >
                <td className="px-3 py-2 font-mono text-gray-400 text-xs">
                  {kv.key}
                </td>
                <td className="px-3 py-2">
                  <span className="px-1.5 py-0.5 rounded bg-gray-800 text-xs text-gray-400 font-mono">
                    {METADATA_TYPE_NAME[kv.valueType] ?? kv.valueType}
                  </span>
                </td>
                <td className="px-3 py-2 font-mono text-gray-200 text-xs max-w-[300px]">
                  {editingKey === kv.key ? (
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => setEditValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter')
                            handleSaveEdit(kv.key, kv.valueType)
                          if (e.key === 'Escape') setEditingKey(null)
                        }}
                        autoFocus
                        className="flex-1 px-2 py-0.5 bg-gray-900 border border-blue-500 rounded text-xs text-gray-200 focus:outline-none"
                      />
                      <button
                        onClick={() => handleSaveEdit(kv.key, kv.valueType)}
                        className="px-2 py-0.5 bg-blue-600 rounded text-xs"
                      >
                        Save
                      </button>
                      <button
                        onClick={() => setEditingKey(null)}
                        className="px-2 py-0.5 bg-gray-700 rounded text-xs"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <span className="truncate block">
                      {formatValue(kv.value, kv.valueType)}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2">
                  {editingKey !== kv.key && kv.valueType !== 9 && (
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => handleEdit(kv.key, kv.value)}
                        className="px-2 py-0.5 text-xs bg-gray-700 hover:bg-gray-600 rounded"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete(kv.key)}
                        className="px-2 py-0.5 text-xs bg-red-900/50 hover:bg-red-800/50 text-red-400 rounded"
                      >
                        Del
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showAdd && <AddMetadataDialog onClose={() => setShowAdd(false)} />}
    </div>
  )
}
