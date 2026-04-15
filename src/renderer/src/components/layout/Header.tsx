import { useFileStore } from '../../store/file-store'
import { formatBytes } from '../../lib/format'

export default function Header() {
  const { fileInfo, isDirty } = useFileStore()

  const handleOpen = async () => {
    const info = await window.api.openFile()
    if (info) useFileStore.getState().setFileInfo(info)
  }

  const handleSave = async () => {
    if (!fileInfo) return
    const result = await window.api.saveFile()
    if (result.success) {
      useFileStore.getState().markClean()
    } else if (result.error && result.error !== 'Cancelled') {
      alert('Save failed: ' + result.error)
    }
  }

  return (
    <header className="flex items-center gap-3 px-4 py-2 bg-gray-900 border-b border-gray-800 shrink-0">
      <button
        onClick={handleOpen}
        className="px-3 py-1.5 text-sm font-medium bg-blue-600 hover:bg-blue-500 rounded-md transition-colors"
      >
        Open
      </button>
      <button
        onClick={handleSave}
        disabled={!fileInfo || !isDirty}
        className="px-3 py-1.5 text-sm font-medium bg-gray-700 hover:bg-gray-600 rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Save
      </button>

      <div className="flex-1" />

      {fileInfo && (
        <div className="flex items-center gap-3 text-sm text-gray-400">
          <span className="font-medium text-gray-200">
            {fileInfo.filePath.split('/').pop()}
            {isDirty && <span className="text-amber-400 ml-1">*</span>}
          </span>
          <span>{formatBytes(fileInfo.fileSize)}</span>
          <span className="px-2 py-0.5 rounded bg-gray-800 text-xs font-mono">
            v{fileInfo.header.version}
          </span>
        </div>
      )}
    </header>
  )
}
