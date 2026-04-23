import { useState, useEffect, useRef, useCallback } from 'react'
import { useFileStore } from '../../store/file-store'
import { formatBytes, formatShape } from '../../lib/format'
import { GGML_TYPE_NAME, getTypeBadgeColor } from '../../lib/tensor-types'
import HexEditor from '../hex-editor/HexEditor'
import DequantizedView from '../tensor-view/DequantizedView'
import TensorStats from '../tensor-view/TensorStats'
import QuantReference from '../tensor-view/QuantReference'
type Tab = 'hex' | 'floats' | 'stats' | 'reference'

export default function TensorDetail() {
  const { fileInfo, selectedTensorIndex, selectTensor, saveVersion } = useFileStore()
  const [activeTab, setActiveTab] = useState<Tab>('hex')
  const [canDeq, setCanDeq] = useState(false)
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [pendingEdits, setPendingEditsRaw] = useState<Map<number, number>>(new Map())
  const [undoStack, setUndoStack] = useState<{ offset: number; oldVal: number | undefined }[]>([])
  const exportRef = useRef<HTMLDivElement>(null)

  // Wrap setPendingEdits to sync to the store (for save + dirty tracking)
  const setPendingEdits = useCallback((edits: Map<number, number>) => {
    setPendingEditsRaw(edits)
    if (selectedTensorIndex !== null) {
      useFileStore.getState().setByteEditsForTensor(selectedTensorIndex, edits)
    }
  }, [selectedTensorIndex])

  useEffect(() => {
    if (selectedTensorIndex === null) return
    window.api.canDequantize(selectedTensorIndex).then(setCanDeq)
    setPendingEditsRaw(new Map())
    setUndoStack([])
  }, [selectedTensorIndex])

  // After save: clear local edit state (saveVersion increments on markClean)
  useEffect(() => {
    if (saveVersion === 0) return
    setPendingEditsRaw(new Map())
    setUndoStack([])
  }, [saveVersion])

  // Close export menu on outside click
  useEffect(() => {
    if (!showExportMenu) return
    const handler = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setShowExportMenu(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showExportMenu])

  if (!fileInfo || selectedTensorIndex === null) return null

  const tensor = fileInfo.tensors[selectedTensorIndex]
  if (!tensor) return null

  const handleExport = async (format: 'raw' | 'npy') => {
    setShowExportMenu(false)
    setExporting(true)
    const result = await window.api.exportTensor(selectedTensorIndex, format)
    setExporting(false)
    if (!result.success && result.error && result.error !== 'Cancelled') {
      alert('Export failed: ' + result.error)
    }
  }

  const tabs: { id: Tab; label: string; disabled?: boolean }[] = [
    { id: 'hex', label: 'Hex View' },
    { id: 'floats', label: 'Float Values', disabled: !canDeq },
    { id: 'stats', label: 'Statistics', disabled: !canDeq },
    { id: 'reference', label: 'Reference' }
  ]

  return (
    <div className="h-full flex flex-col">
      {/* Tensor header */}
      <div className="p-4 border-b border-gray-800 shrink-0">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-mono text-sm font-bold text-gray-200 truncate pr-4">
            {tensor.name}
          </h3>
          <div className="flex items-center gap-2 shrink-0">
            {/* Export button */}
            <div className="relative" ref={exportRef}>
              <button
                onClick={() => setShowExportMenu(!showExportMenu)}
                disabled={exporting}
                className="text-gray-500 hover:text-emerald-400 text-xs px-2 py-1 rounded hover:bg-gray-800 transition-colors disabled:opacity-50"
              >
                {exporting ? 'Exporting...' : 'Export'}
              </button>
              {showExportMenu && (
                <div className="absolute right-0 top-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 py-1 w-56">
                  <button
                    onClick={() => handleExport('raw')}
                    className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors"
                  >
                    <div className="font-medium">Raw binary (.bin)</div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      Exact bytes as stored ({GGML_TYPE_NAME[tensor.type] ?? 'unknown'} format)
                    </div>
                  </button>
                  <button
                    onClick={() => handleExport('npy')}
                    disabled={!canDeq}
                    className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <div className="font-medium">NumPy array (.npy)</div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      Dequantized float32 values, loadable with np.load()
                    </div>
                  </button>
                </div>
              )}
            </div>
            <button
              onClick={() => useFileStore.getState().showTensorInDiagram(tensor.name)}
              className="text-gray-500 hover:text-blue-400 text-xs px-2 py-1 rounded hover:bg-gray-800 transition-colors"
              title="Show in Diagram"
            >
              Diagram
            </button>
            <button
              onClick={() => selectTensor(null)}
              className="text-gray-500 hover:text-gray-300 text-sm px-2"
            >
              Close
            </button>
          </div>
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
          <HexEditor
            tensorIndex={selectedTensorIndex}
            sizeBytes={tensor.sizeBytes}
            tensorType={tensor.type}
            pendingEdits={pendingEdits}
            onPendingEditsChange={setPendingEdits}
            undoStack={undoStack}
            onUndoStackChange={setUndoStack}
          />
        )}
        {activeTab === 'floats' && canDeq && (
          <DequantizedView tensorIndex={selectedTensorIndex} tensor={tensor} pendingEdits={pendingEdits} />
        )}
        {activeTab === 'stats' && canDeq && (
          <TensorStats tensorIndex={selectedTensorIndex} />
        )}
        {activeTab === 'reference' && <QuantReference type={tensor.type} />}
      </div>
    </div>
  )
}
