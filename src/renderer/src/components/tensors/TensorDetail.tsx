import { useState, useEffect } from 'react'
import { useFileStore } from '../../store/file-store'
import { formatBytes, formatShape } from '../../lib/format'
import { GGML_TYPE_NAME, getTypeBadgeColor } from '../../lib/tensor-types'
import HexEditor from '../hex-editor/HexEditor'
import DequantizedView from '../tensor-view/DequantizedView'
import TensorStats from '../tensor-view/TensorStats'

type Tab = 'hex' | 'floats' | 'stats'

export default function TensorDetail() {
  const { fileInfo, selectedTensorIndex, selectTensor } = useFileStore()
  const [activeTab, setActiveTab] = useState<Tab>('hex')
  const [canDeq, setCanDeq] = useState(false)

  useEffect(() => {
    if (selectedTensorIndex === null) return
    window.api.canDequantize(selectedTensorIndex).then(setCanDeq)
  }, [selectedTensorIndex])

  if (!fileInfo || selectedTensorIndex === null) return null

  const tensor = fileInfo.tensors[selectedTensorIndex]
  if (!tensor) return null

  const tabs: { id: Tab; label: string; disabled?: boolean }[] = [
    { id: 'hex', label: 'Hex View' },
    { id: 'floats', label: 'Float Values', disabled: !canDeq },
    { id: 'stats', label: 'Statistics', disabled: !canDeq }
  ]

  return (
    <div className="h-full flex flex-col">
      {/* Tensor header */}
      <div className="p-4 border-b border-gray-800 shrink-0">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-mono text-sm font-bold text-gray-200 truncate pr-4">
            {tensor.name}
          </h3>
          <button
            onClick={() => selectTensor(null)}
            className="text-gray-500 hover:text-gray-300 text-sm px-2"
          >
            Close
          </button>
        </div>
        <div className="flex items-center gap-3 text-xs text-gray-400">
          <span
            className={`px-1.5 py-0.5 rounded font-mono ${getTypeBadgeColor(tensor.type)}`}
          >
            {GGML_TYPE_NAME[tensor.type] ?? `?${tensor.type}`}
          </span>
          <span>Shape: {formatShape(tensor.dims)}</span>
          <span>{formatBytes(tensor.sizeBytes)}</span>
          <span>{tensor.nDims}D</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-800 shrink-0">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => !tab.disabled && setActiveTab(tab.id)}
            disabled={tab.disabled}
            className={`px-4 py-2 text-sm font-medium transition-colors
              ${
                activeTab === tab.id
                  ? 'border-b-2 border-blue-500 text-blue-400'
                  : 'text-gray-500 hover:text-gray-300'
              }
              ${tab.disabled ? 'opacity-30 cursor-not-allowed' : ''}`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'hex' && (
          <HexEditor tensorIndex={selectedTensorIndex} sizeBytes={tensor.sizeBytes} />
        )}
        {activeTab === 'floats' && canDeq && (
          <DequantizedView tensorIndex={selectedTensorIndex} tensor={tensor} />
        )}
        {activeTab === 'stats' && canDeq && (
          <TensorStats tensorIndex={selectedTensorIndex} />
        )}
      </div>
    </div>
  )
}
