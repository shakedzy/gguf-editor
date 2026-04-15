import { ipcMain, BrowserWindow } from 'electron'
import {
  getCurrentFileInfo,
  openFileDialog,
  saveFileDialog,
  updateMetadataInMemory,
  addMetadataInMemory,
  deleteMetadataInMemory
} from './file-manager'
import { readTensorChunk, dequantizeTensorChunk, computeTensorStats } from './gguf/tensor-data'
import { writeGgufFile } from './gguf/writer'
import { canDequantize } from './gguf/dequantize'

function serializeFileInfo(info: any): any {
  return JSON.parse(
    JSON.stringify(info, (_key, value) =>
      typeof value === 'bigint' ? value.toString() : value
    )
  )
}

export function registerIpcHandlers(): void {
  ipcMain.handle('gguf:open', async (event) => {
    const window = BrowserWindow.fromWebContents(event.sender)
    if (!window) return null
    const info = await openFileDialog(window)
    if (!info) return null
    return serializeFileInfo(info)
  })

  ipcMain.handle('gguf:get-file-info', () => {
    const info = getCurrentFileInfo()
    if (!info) return null
    return serializeFileInfo(info)
  })

  ipcMain.handle(
    'gguf:read-tensor-chunk',
    (_event, tensorIndex: number, chunkOffset: number, chunkLength: number) => {
      const info = getCurrentFileInfo()
      if (!info || tensorIndex >= info.tensors.length) return null
      const tensor = info.tensors[tensorIndex]
      const buf = readTensorChunk(
        info.filePath,
        info.dataStartOffset,
        tensor,
        chunkOffset,
        chunkLength
      )
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
    }
  )

  ipcMain.handle(
    'gguf:dequantize-tensor',
    (_event, tensorIndex: number, chunkOffset: number, count: number) => {
      const info = getCurrentFileInfo()
      if (!info || tensorIndex >= info.tensors.length) return null
      const tensor = info.tensors[tensorIndex]
      const floats = dequantizeTensorChunk(
        info.filePath,
        info.dataStartOffset,
        tensor,
        chunkOffset,
        count
      )
      if (!floats) return null
      return floats.buffer.slice(
        floats.byteOffset,
        floats.byteOffset + floats.byteLength
      )
    }
  )

  ipcMain.handle('gguf:can-dequantize', (_event, tensorIndex: number) => {
    const info = getCurrentFileInfo()
    if (!info || tensorIndex >= info.tensors.length) return false
    return canDequantize(info.tensors[tensorIndex].type)
  })

  ipcMain.handle('gguf:tensor-stats', (_event, tensorIndex: number) => {
    const info = getCurrentFileInfo()
    if (!info || tensorIndex >= info.tensors.length) return null
    return computeTensorStats(
      info.filePath,
      info.dataStartOffset,
      info.tensors[tensorIndex]
    )
  })

  ipcMain.handle(
    'gguf:update-metadata',
    (_event, key: string, value: any, valueType: number) => {
      updateMetadataInMemory(key, value, valueType)
      return true
    }
  )

  ipcMain.handle(
    'gguf:add-metadata',
    (_event, key: string, value: any, valueType: number) => {
      addMetadataInMemory(key, value, valueType)
      return true
    }
  )

  ipcMain.handle('gguf:delete-metadata', (_event, key: string) => {
    deleteMetadataInMemory(key)
    return true
  })

  ipcMain.handle('gguf:save', async (event, filePath?: string) => {
    const info = getCurrentFileInfo()
    if (!info) return { success: false, error: 'No file loaded' }

    try {
      let outPath = filePath
      if (!outPath) {
        const window = BrowserWindow.fromWebContents(event.sender)
        if (!window) return { success: false, error: 'No window' }
        outPath = await saveFileDialog(window)
        if (!outPath) return { success: false, error: 'Cancelled' }
      }

      await writeGgufFile(info, outPath)
      return { success: true, filePath: outPath }
    } catch (err: any) {
      return { success: false, error: err.message }
    }
  })
}
