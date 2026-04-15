import { useFileStore } from '../../store/file-store'
import FileOverview from '../overview/FileOverview'
import MetadataEditor from '../metadata/MetadataEditor'
import TensorList from '../tensors/TensorList'
import TensorDetail from '../tensors/TensorDetail'

export default function MainPanel() {
  const { fileInfo, activeView, selectedTensorIndex } = useFileStore()

  if (!fileInfo) {
    return (
      <main className="flex-1 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="text-6xl mb-4 font-bold text-gray-700">GGUF</div>
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

  return (
    <main className="flex-1 overflow-hidden flex">
      <div className={`flex-1 overflow-auto ${selectedTensorIndex !== null ? 'w-1/2' : ''}`}>
        {activeView === 'overview' && <FileOverview />}
        {activeView === 'metadata' && <MetadataEditor />}
        {activeView === 'tensors' && <TensorList />}
      </div>

      {activeView === 'tensors' && selectedTensorIndex !== null && (
        <div className="w-1/2 border-l border-gray-800 overflow-hidden">
          <TensorDetail />
        </div>
      )}
    </main>
  )
}
