import { useRef, useState, useCallback, useEffect } from 'react'
import { useFileStore } from '../../store/file-store'
import FileOverview from '../overview/FileOverview'
import MetadataEditor from '../metadata/MetadataEditor'
import TensorList from '../tensors/TensorList'
import TensorDetail from '../tensors/TensorDetail'
import ModelDiagram from '../diagram/ModelDiagram'

export default function MainPanel() {
  const { fileInfo, activeView, selectedTensorIndex } = useFileStore()
  const [splitPercent, setSplitPercent] = useState(40)
  const containerRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
  }, [])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const pct = ((e.clientX - rect.left) / rect.width) * 100
      setSplitPercent(Math.max(20, Math.min(80, pct)))
    }
    const handleMouseUp = () => {
      dragging.current = false
    }
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [])

  if (!fileInfo) {
    return (
      <main className="flex-1 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="text-6xl mb-4 font-bold text-gray-700">GGUF Editor</div>
          <p className="text-lg mb-2">No file loaded</p>
          <p className="text-sm">
            Press{' '}
            <kbd className="px-1.5 py-0.5 rounded bg-gray-800 text-gray-300 text-xs font-mono">
              Cmd+O
            </kbd>{' '}
            or click <strong>Open</strong> to load a GGUF file
          </p>
        </div>
      </main>
    )
  }

  const showDetail = activeView === 'tensors' && selectedTensorIndex !== null

  return (
    <main ref={containerRef} className="flex-1 overflow-hidden flex">
      <div
        className="overflow-auto"
        style={{ width: showDetail ? `${splitPercent}%` : '100%' }}
      >
        {activeView === 'overview' && <FileOverview />}
        {activeView === 'metadata' && <MetadataEditor />}
        {activeView === 'tensors' && <TensorList />}
        {activeView === 'diagram' && <ModelDiagram />}
      </div>

      {showDetail && (
        <>
          {/* Drag handle */}
          <div
            onMouseDown={handleMouseDown}
            className="w-1 bg-gray-800 hover:bg-blue-600 cursor-col-resize shrink-0 transition-colors"
          />
          <div
            className="overflow-hidden"
            style={{ width: `${100 - splitPercent}%` }}
          >
            <TensorDetail />
          </div>
        </>
      )}
    </main>
  )
}
