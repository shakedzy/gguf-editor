import { contextBridge, ipcRenderer } from 'electron'

export interface GgufApi {
  getAppVersion: () => Promise<string>
  getArchRegistry: () => Promise<{
    tensorRoles: Record<string, string>
    knownArchitectures: string[]
    fetchedAt: string
    source: 'live' | 'cached' | 'offline'
  }>
  refreshArchRegistry: () => Promise<{
    tensorRoles: Record<string, string>
    knownArchitectures: string[]
    fetchedAt: string
    source: 'live' | 'cached' | 'offline'
  }>
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
    count: number,
    edits?: [number, number][]
  ) => Promise<ArrayBuffer | null>
  canDequantize: (tensorIndex: number) => Promise<boolean>
  getTensorStats: (tensorIndex: number) => Promise<any>
  updateMetadata: (key: string, value: any, valueType: number) => Promise<boolean>
  addMetadata: (key: string, value: any, valueType: number) => Promise<boolean>
  deleteMetadata: (key: string) => Promise<boolean>
  exportTensor: (tensorIndex: number, format: 'raw' | 'npy') => Promise<{ success: boolean; error?: string; filePath?: string }>
  saveFile: (filePath?: string, byteEdits?: Record<number, [number, number][]>) => Promise<{ success: boolean; error?: string; filePath?: string }>
  onMenuOpen: (callback: () => void) => void
  onMenuSave: (callback: () => void) => void
  onMenuSaveAs: (callback: () => void) => void
  onSaveProgress: (callback: (fraction: number) => void) => void
}

const api: GgufApi = {
  getAppVersion: () => ipcRenderer.invoke('app:version'),
  getArchRegistry: () => ipcRenderer.invoke('app:arch-registry'),
  refreshArchRegistry: () => ipcRenderer.invoke('app:arch-registry-refresh'),
  openFile: () => ipcRenderer.invoke('gguf:open'),
  getFileInfo: () => ipcRenderer.invoke('gguf:get-file-info'),
  readTensorChunk: (tensorIndex, offset, length) =>
    ipcRenderer.invoke('gguf:read-tensor-chunk', tensorIndex, offset, length),
  dequantizeTensor: (tensorIndex, offset, count, edits?) =>
    ipcRenderer.invoke('gguf:dequantize-tensor', tensorIndex, offset, count, edits),
  canDequantize: (tensorIndex) =>
    ipcRenderer.invoke('gguf:can-dequantize', tensorIndex),
  getTensorStats: (tensorIndex) =>
    ipcRenderer.invoke('gguf:tensor-stats', tensorIndex),
  updateMetadata: (key, value, valueType) =>
    ipcRenderer.invoke('gguf:update-metadata', key, value, valueType),
  addMetadata: (key, value, valueType) =>
    ipcRenderer.invoke('gguf:add-metadata', key, value, valueType),
  deleteMetadata: (key) => ipcRenderer.invoke('gguf:delete-metadata', key),
  exportTensor: (tensorIndex, format) =>
    ipcRenderer.invoke('gguf:export-tensor', tensorIndex, format),
  saveFile: (filePath?, byteEdits?) => ipcRenderer.invoke('gguf:save', filePath, byteEdits),
  onMenuOpen: (cb) => ipcRenderer.on('menu:open', cb),
  onMenuSave: (cb) => ipcRenderer.on('menu:save', cb),
  onMenuSaveAs: (cb) => ipcRenderer.on('menu:save-as', cb),
  onSaveProgress: (cb) => ipcRenderer.on('save:progress', (_e, fraction) => cb(fraction))
}

contextBridge.exposeInMainWorld('api', api)
