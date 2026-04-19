import { app, ipcMain, BrowserWindow, dialog } from 'electron'
import {
  getCurrentFileInfo,
  openFileDialog,
  saveFileDialog,
  updateMetadataInMemory,
  addMetadataInMemory,
  deleteMetadataInMemory
} from './file-manager'
import { readTensorChunk, dequantizeTensorChunk, computeTensorStats, exportTensorRaw, exportTensorNumpy } from './gguf/tensor-data'
import { getArchRegistry, refreshArchRegistry } from './arch-registry'
import { writeGgufFile } from './gguf/writer'
import { canDequantize } from './gguf/dequantize'
import { GGML_TYPE_NAME } from './gguf/constants'

function serializeFileInfo(info: any): any {
  return JSON.parse(
    JSON.stringify(info, (_key, value) =>
      typeof value === 'bigint' ? value.toString() : value
    )
  )
}

export function registerIpcHandlers(): void {
  ipcMain.handle('app:version', () => app.getVersion())
  ipcMain.handle('app:arch-registry', () => getArchRegistry())
  ipcMain.handle('app:arch-registry-refresh', () => refreshArchRegistry())

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
    (_event, tensorIndex: number, chunkOffset: number, count: number, edits?: [number, number][]) => {
      const info = getCurrentFileInfo()
      if (!info || tensorIndex >= info.tensors.length) return null
      const tensor = info.tensors[tensorIndex]
      const floats = dequantizeTensorChunk(
        info.filePath,
        info.dataStartOffset,
        tensor,
        chunkOffset,
        count,
        edits
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

  ipcMain.handle(
    'gguf:export-tensor',
    async (event, tensorIndex: number, format: 'raw' | 'npy') => {
      const info = getCurrentFileInfo()
      if (!info || tensorIndex >= info.tensors.length) {
        return { success: false, error: 'No file or invalid tensor' }
      }
      const tensor = info.tensors[tensorIndex]
      const window = BrowserWindow.fromWebContents(event.sender)
      if (!window) return { success: false, error: 'No window' }

      const typeName = GGML_TYPE_NAME[tensor.type] ?? 'unknown'
      const safeName = tensor.name.replace(/[^a-zA-Z0-9._-]/g, '_')

      const ext = format === 'npy' ? 'npy' : 'bin'
      const result = await dialog.showSaveDialog(window, {
        title: `Export Tensor: ${tensor.name}`,
        defaultPath: `${safeName}.${ext}`,
        filters:
          format === 'npy'
            ? [{ name: 'NumPy Array', extensions: ['npy'] }]
            : [
                { name: 'Raw Binary', extensions: ['bin'] },
                { name: 'All Files', extensions: ['*'] }
              ]
      })

      if (result.canceled || !result.filePath) {
        return { success: false, error: 'Cancelled' }
      }

      try {
        if (format === 'npy') {
          const ok = exportTensorNumpy(
            info.filePath,
            info.dataStartOffset,
            tensor,
            result.filePath
          )
          if (!ok) return { success: false, error: `Cannot dequantize ${typeName} to float32` }
        } else {
          exportTensorRaw(
            info.filePath,
            info.dataStartOffset,
            tensor,
            result.filePath
          )
        }
        return { success: true, filePath: result.filePath }
      } catch (err: any) {
        return { success: false, error: err.message }
      }
    }
  )

  ipcMain.handle('gguf:save', async (event, filePath?: string, byteEdits?: Record<number, [number, number][]>) => {
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

      const sender = event.sender
      await writeGgufFile(info, outPath, byteEdits, (fraction) => {
        sender.send('save:progress', fraction)
      })
      return { success: true, filePath: outPath }
    } catch (err: any) {
      return { success: false, error: err.message }
    }
  })
}
