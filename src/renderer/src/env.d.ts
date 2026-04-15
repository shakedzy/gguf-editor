/// <reference types="vite/client" />

interface GgufApi {
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
  saveFile: (filePath?: string) => Promise<{
    success: boolean
    error?: string
    filePath?: string
  }>
  onMenuOpen: (callback: () => void) => void
  onMenuSave: (callback: () => void) => void
  onMenuSaveAs: (callback: () => void) => void
}

interface Window {
  api: GgufApi
}
