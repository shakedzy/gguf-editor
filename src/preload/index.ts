import { contextBridge, ipcRenderer } from 'electron'

export interface GgufApi {
  openFile: () => Promise<any>
  getFileInfo: () => Promise<any>
  readTensorChunk: (
    tensorIndex: number,
    offset: number,
    length: number
  ) => Promise<ArrayBuffer | null>
  dequantizeTensor: (
    tensorIndex: number,
    offset: number,
    count: number
  ) => Promise<ArrayBuffer | null>
  canDequantize: (tensorIndex: number) => Promise<boolean>
  getTensorStats: (tensorIndex: number) => Promise<any>
  updateMetadata: (key: string, value: any, valueType: number) => Promise<boolean>
  addMetadata: (key: string, value: any, valueType: number) => Promise<boolean>
  deleteMetadata: (key: string) => Promise<boolean>
  saveFile: (filePath?: string) => Promise<{ success: boolean; error?: string; filePath?: string }>
  onMenuOpen: (callback: () => void) => void
  onMenuSave: (callback: () => void) => void
  onMenuSaveAs: (callback: () => void) => void
}

const api: GgufApi = {
  openFile: () => ipcRenderer.invoke('gguf:open'),
  getFileInfo: () => ipcRenderer.invoke('gguf:get-file-info'),
  readTensorChunk: (tensorIndex, offset, length) =>
    ipcRenderer.invoke('gguf:read-tensor-chunk', tensorIndex, offset, length),
  dequantizeTensor: (tensorIndex, offset, count) =>
    ipcRenderer.invoke('gguf:dequantize-tensor', tensorIndex, offset, count),
  canDequantize: (tensorIndex) =>
    ipcRenderer.invoke('gguf:can-dequantize', tensorIndex),
  getTensorStats: (tensorIndex) =>
    ipcRenderer.invoke('gguf:tensor-stats', tensorIndex),
  updateMetadata: (key, value, valueType) =>
    ipcRenderer.invoke('gguf:update-metadata', key, value, valueType),
  addMetadata: (key, value, valueType) =>
    ipcRenderer.invoke('gguf:add-metadata', key, value, valueType),
  deleteMetadata: (key) => ipcRenderer.invoke('gguf:delete-metadata', key),
  saveFile: (filePath?) => ipcRenderer.invoke('gguf:save', filePath),
  onMenuOpen: (cb) => ipcRenderer.on('menu:open', cb),
  onMenuSave: (cb) => ipcRenderer.on('menu:save', cb),
  onMenuSaveAs: (cb) => ipcRenderer.on('menu:save-as', cb)
}

contextBridge.exposeInMainWorld('api', api)
