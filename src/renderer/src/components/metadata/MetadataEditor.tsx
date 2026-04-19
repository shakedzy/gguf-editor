import { useState } from 'react'
import { useFileStore } from '../../store/file-store'
import { METADATA_TYPE_NAME } from '../../lib/tensor-types'
import { getMetadataTooltip } from '../../lib/tooltips'
import Tooltip from '../ui/Tooltip'
import AddMetadataDialog from './AddMetadataDialog'
import ArrayViewer from './ArrayViewer'

function formatValue(value: any, valueType: number): string {
  if (valueType === 9 && value && typeof value === 'object' && value.values) {
    const arr = value.values as any[]
    if (arr.length <= 5) return `[${arr.join(', ')}]`
    return `[${arr.slice(0, 3).join(', ')}, ... (${arr.length.toLocaleString()} items)]`
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

interface ArrayViewState {
  key: string
  arrayType: number
  values: any[]
}

export default function MetadataEditor() {
  const { fileInfo } = useFileStore()
  const [search, setSearch] = useState('')
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [showAdd, setShowAdd] = useState(false)
  const [arrayView, setArrayView] = useState<ArrayViewState | null>(null)

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

  const handleOpenArray = (kv: { key: string; valueType: number; value: any }) => {
    if (kv.valueType === 9 && kv.value && typeof kv.value === 'object') {
      setArrayView({
        key: kv.key,
        arrayType: kv.value.type,
        values: kv.value.values
      })
    }
  }

  const handleSaveArray = async (key: string, arrayType: number, newValues: any[]) => {
    const newValue = { type: arrayType, values: newValues }
    await window.api.updateMetadata(key, newValue, 9)
    useFileStore.getState().updateMetadata(key, newValue, 9)
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
              <th className="px-3 py-2 w-28" />
            </tr>
          </thead>
          <tbody>
            {filtered.map((kv) => {
              const tip = getMetadataTooltip(kv.key)
              const isArray = kv.valueType === 9

              return (
                <tr
                  key={kv.key}
                  className="border-b border-gray-800/50 hover:bg-gray-800/30 group"
                >
                  <td className="px-3 py-2 font-mono text-gray-400 text-xs">
                    {tip ? (
                      <Tooltip text={tip}>
                        <span className="border-b border-dotted border-gray-600">
                          {kv.key}
                        </span>
                        <span className="ml-1 text-gray-600">?</span>
                      </Tooltip>
                    ) : (
                      kv.key
                    )}
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
                    {editingKey !== kv.key && (
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        {isArray ? (
                          <button
                            onClick={() => handleOpenArray(kv)}
                            className="px-2 py-0.5 text-xs bg-purple-900/50 hover:bg-purple-800/50 text-purple-400 rounded"
                          >
                            View
                          </button>
                        ) : (
                          <button
                            onClick={() => handleEdit(kv.key, kv.value)}
                            className="px-2 py-0.5 text-xs bg-gray-700 hover:bg-gray-600 rounded"
                          >
                            Edit
                          </button>
                        )}
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
              )
            })}
          </tbody>
        </table>
      </div>

      {showAdd && <AddMetadataDialog onClose={() => setShowAdd(false)} />}

      {arrayView && (
        <ArrayViewer
          metadataKey={arrayView.key}
          arrayType={arrayView.arrayType}
          values={arrayView.values}
          onClose={() => setArrayView(null)}
          onSave={(newValues) =>
            handleSaveArray(arrayView.key, arrayView.arrayType, newValues)
          }
        />
      )}
    </div>
  )
}
