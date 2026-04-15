import { useState } from 'react'
import { useFileStore } from '../../store/file-store'

const TYPES = [
  { label: 'STRING', value: 8 },
  { label: 'UINT32', value: 4 },
  { label: 'INT32', value: 5 },
  { label: 'FLOAT32', value: 6 },
  { label: 'BOOL', value: 7 },
  { label: 'UINT8', value: 0 },
  { label: 'INT8', value: 1 },
  { label: 'UINT16', value: 2 },
  { label: 'INT16', value: 3 },
  { label: 'UINT64', value: 10 },
  { label: 'INT64', value: 11 },
  { label: 'FLOAT64', value: 12 }
]

interface Props {
  onClose: () => void
}

export default function AddMetadataDialog({ onClose }: Props) {
  const [key, setKey] = useState('')
  const [valueType, setValueType] = useState(8)
  const [rawValue, setRawValue] = useState('')

  const handleAdd = async () => {
    if (!key.trim()) return

    let parsed: any = rawValue
    if ([0, 1, 2, 3, 4, 5, 10, 11].includes(valueType)) {
      parsed = Number(rawValue)
      if (isNaN(parsed)) return
    } else if (valueType === 6 || valueType === 12) {
      parsed = parseFloat(rawValue)
      if (isNaN(parsed)) return
    } else if (valueType === 7) {
      parsed = rawValue === 'true' || rawValue === '1'
    }

    await window.api.addMetadata(key, parsed, valueType)
    useFileStore.getState().addMetadata(key, parsed, valueType)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-6 w-[400px] space-y-4">
        <h3 className="text-lg font-bold text-gray-100">Add Metadata</h3>

        <div>
          <label className="block text-xs text-gray-500 mb-1">Key</label>
          <input
            type="text"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="e.g. general.name"
            className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded text-sm text-gray-200 focus:outline-none focus:border-blue-500"
          />
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">Type</label>
          <select
            value={valueType}
            onChange={(e) => setValueType(Number(e.target.value))}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded text-sm text-gray-200 focus:outline-none focus:border-blue-500"
          >
            {TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">Value</label>
          <input
            type="text"
            value={rawValue}
            onChange={(e) => setRawValue(e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded text-sm text-gray-200 focus:outline-none focus:border-blue-500"
          />
        </div>

        <div className="flex justify-end gap-2 pt-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-gray-700 hover:bg-gray-600 rounded-md"
          >
            Cancel
          </button>
          <button
            onClick={handleAdd}
            disabled={!key.trim()}
            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded-md disabled:opacity-40"
          >
            Add
          </button>
        </div>
      </div>
    </div>
  )
}
