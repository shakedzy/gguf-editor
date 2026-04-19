import { useState, useEffect } from 'react'
import { useFileStore, ActiveView } from '../../store/file-store'
import { formatNumber } from '../../lib/format'

const NAV_ITEMS: { view: ActiveView; label: string; icon: string }[] = [
  { view: 'overview', label: 'Overview', icon: 'i' },
  { view: 'metadata', label: 'Metadata', icon: 'M' },
  { view: 'tensors', label: 'Tensors', icon: 'T' },
  { view: 'diagram', label: 'Diagram', icon: 'D' }
]

export default function Sidebar() {
  const { fileInfo, activeView, setActiveView } = useFileStore()
  const [appVersion, setAppVersion] = useState('')

  useEffect(() => {
    window.api.getAppVersion().then(setAppVersion)
  }, [])

  return (
    <aside className="w-52 bg-gray-900/50 border-r border-gray-800 flex flex-col shrink-0">
      <nav className="flex flex-col gap-1 p-2">
        {NAV_ITEMS.map(({ view, label, icon }) => (
          <button
            key={view}
            onClick={() => setActiveView(view)}
            disabled={!fileInfo}
            className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors text-left
              ${
                activeView === view
                  ? 'bg-blue-600/20 text-blue-400'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }
              disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            <span className="w-6 h-6 rounded bg-gray-800 flex items-center justify-center text-xs font-bold">
              {icon}
            </span>
            {label}
          </button>
        ))}
      </nav>

      {fileInfo && (
        <div className="mt-auto p-3 border-t border-gray-800 text-xs text-gray-500 space-y-1">
          <div>
            <span className="text-gray-400">{formatNumber(fileInfo.header.tensorCount)}</span>{' '}
            tensors
          </div>
          <div>
            <span className="text-gray-400">{formatNumber(fileInfo.header.metadataKvCount)}</span>{' '}
            metadata keys
          </div>
          <div>
            Alignment: <span className="text-gray-400">{fileInfo.alignment}</span>
          </div>
        </div>
      )}

      {!fileInfo && <div className="flex-1" />}

      {appVersion && (
        <div className="p-3 border-t border-gray-800 text-[10px] text-gray-600">
          GGUF Editor v{appVersion}
        </div>
      )}
    </aside>
  )
}
