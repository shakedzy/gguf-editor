import { useState, useEffect } from 'react'
import { useFileStore } from '../../store/file-store'
import { formatBytes } from '../../lib/format'

export default function Header() {
  const { fileInfo, isDirty } = useFileStore()
  const [saving, setSaving] = useState(false)
  const [saveProgress, setSaveProgress] = useState(0)

  useEffect(() => {
    window.api.onSaveProgress((fraction) => {
      setSaveProgress(fraction)
    })
  }, [])

  const handleOpen = async () => {
    const info = await window.api.openFile()
    if (info) useFileStore.getState().setFileInfo(info)
  }

  const doSave = async (filePath?: string) => {
    if (!fileInfo) return
    setSaving(true)
    setSaveProgress(0)
    const edits = useFileStore.getState().byteEdits
    const result = await window.api.saveFile(filePath, edits)
    setSaving(false)
    if (result.success) {
      useFileStore.getState().markClean()
    } else if (result.error && result.error !== 'Cancelled') {
      alert('Save failed: ' + result.error)
    }
  }

  return (
    <header className="shrink-0">
      <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 border-b border-gray-800">
        <button
          onClick={handleOpen}
          disabled={saving}
          className="px-3 py-1.5 text-sm font-medium bg-blue-600 hover:bg-blue-500 rounded-md transition-colors disabled:opacity-40"
        >
          Open
        </button>
        <button
          onClick={() => doSave(fileInfo?.filePath)}
          disabled={!fileInfo || !isDirty || saving}
          className="px-3 py-1.5 text-sm font-medium bg-gray-700 hover:bg-gray-600 rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {saving ? `Saving ${Math.round(saveProgress * 100)}%` : 'Save'}
        </button>
        <button
          onClick={() => doSave(undefined)}
          disabled={!fileInfo || saving}
          className="px-3 py-1.5 text-sm font-medium bg-gray-700 hover:bg-gray-600 rounded-md transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Save As...
        </button>

        <div className="flex-1" />

        {fileInfo && (
          <div className="flex items-center gap-3 text-sm text-gray-400">
            {(() => {
              const modelName = fileInfo.metadata.find((m) => m.key === 'general.name')?.value
              return modelName ? (
                <span className="text-gray-300 font-medium truncate max-w-[200px]">{String(modelName)}</span>
              ) : null
            })()}
            <span className="font-medium text-gray-200">
              {fileInfo.displayName || fileInfo.filePath.split('/').pop()}
              {isDirty && <span className="text-amber-400 ml-1">*</span>}
            </span>
            <span>{formatBytes(fileInfo.fileSize)}</span>
            <span className="px-2 py-0.5 rounded bg-gray-800 text-xs font-mono">
              v{fileInfo.header.version}
            </span>
          </div>
        )}
      </div>
    </header>
  )
}
