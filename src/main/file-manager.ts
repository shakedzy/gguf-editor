import { dialog, BrowserWindow } from 'electron'
import { GgufFileInfo } from './gguf/types'
import { parseGgufFile } from './gguf/parser'

let currentFileInfo: GgufFileInfo | null = null

export function getCurrentFileInfo(): GgufFileInfo | null {
  return currentFileInfo
}

export function setCurrentFileInfo(info: GgufFileInfo | null): void {
  currentFileInfo = info
}

export function updateMetadataInMemory(
  key: string,
  value: any,
  valueType: number
): void {
  if (!currentFileInfo) return
  const kv = currentFileInfo.metadata.find((m) => m.key === key)
  if (kv) {
    kv.value = value
    kv.valueType = valueType
  }
}

export function addMetadataInMemory(
  key: string,
  value: any,
  valueType: number
): void {
  if (!currentFileInfo) return
  currentFileInfo.metadata.push({ key, valueType, value })
  currentFileInfo.header.metadataKvCount = currentFileInfo.metadata.length
}

export function deleteMetadataInMemory(key: string): void {
  if (!currentFileInfo) return
  currentFileInfo.metadata = currentFileInfo.metadata.filter(
    (m) => m.key !== key
  )
  currentFileInfo.header.metadataKvCount = currentFileInfo.metadata.length
}

export async function openFileDialog(
  window: BrowserWindow
): Promise<GgufFileInfo | null> {
  const result = await dialog.showOpenDialog(window, {
    title: 'Open GGUF File',
    filters: [
      { name: 'GGUF Files', extensions: ['gguf'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    properties: ['openFile']
  })

  if (result.canceled || result.filePaths.length === 0) return null

  const filePath = result.filePaths[0]
  const fileInfo = await parseGgufFile(filePath)
  currentFileInfo = fileInfo
  return fileInfo
}

export async function saveFileDialog(
  window: BrowserWindow
): Promise<string | null> {
  const result = await dialog.showSaveDialog(window, {
    title: 'Save GGUF File',
    filters: [
      { name: 'GGUF Files', extensions: ['gguf'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  })

  if (result.canceled || !result.filePath) return null
  return result.filePath
}
