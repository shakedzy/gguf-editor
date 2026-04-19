import { useEffect } from 'react'
import { useFileStore } from './store/file-store'
import Header from './components/layout/Header'
import Sidebar from './components/layout/Sidebar'
import MainPanel from './components/layout/MainPanel'

export default function App() {
  const { setFileInfo, fileInfo } = useFileStore()

  useEffect(() => {
    window.api.onMenuOpen(async () => {
      const info = await window.api.openFile()
      if (info) setFileInfo(info)
    })

    window.api.onMenuSave(async () => {
      if (!fileInfo) return
      const edits = useFileStore.getState().byteEdits
      const result = await window.api.saveFile(fileInfo.filePath, edits)
      if (result.success) {
        useFileStore.getState().markClean()
      }
    })

    window.api.onMenuSaveAs(async () => {
      if (!fileInfo) return
      const edits = useFileStore.getState().byteEdits
      const result = await window.api.saveFile(undefined, edits)
      if (result.success) {
        useFileStore.getState().markClean()
      }
    })
  }, [])

  return (
    <div className="flex flex-col h-screen bg-gray-950">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <MainPanel />
      </div>
    </div>
  )
}
