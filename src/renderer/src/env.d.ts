/// <reference types="vite/client" />

interface GgufApi {
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
  exportTensor: (tensorIndex: number, format: 'raw' | 'npy') => Promise<{
    success: boolean
    error?: string
    filePath?: string
  }>
  saveFile: (filePath?: string, byteEdits?: Record<number, [number, number][]>) => Promise<{
    success: boolean
    error?: string
    filePath?: string
  }>
  onMenuOpen: (callback: () => void) => void
  onMenuSave: (callback: () => void) => void
  onMenuSaveAs: (callback: () => void) => void
  onSaveProgress: (callback: (fraction: number) => void) => void
}

interface Window {
  api: GgufApi
}
