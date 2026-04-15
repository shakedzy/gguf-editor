import { create } from 'zustand'

export interface GgufHeader {
  magic: number
  version: number
  tensorCount: number
  metadataKvCount: number
}

export interface GgufMetadataKV {
  key: string
  valueType: number
  value: any
}

export interface GgufTensorInfo {
  name: string
  nDims: number
  dims: number[]
  type: number
  offset: number
  sizeBytes: number
}

export interface GgufFileInfo {
  header: GgufHeader
  metadata: GgufMetadataKV[]
  tensors: GgufTensorInfo[]
  alignment: number
  dataStartOffset: number
  fileSize: number
  filePath: string
}

export type ActiveView = 'overview' | 'metadata' | 'tensors'

interface FileStore {
  fileInfo: GgufFileInfo | null
  isDirty: boolean
  activeView: ActiveView
  selectedTensorIndex: number | null

  setFileInfo: (info: GgufFileInfo | null) => void
  setActiveView: (view: ActiveView) => void
  selectTensor: (index: number | null) => void
  markDirty: () => void
  markClean: () => void
  updateMetadata: (key: string, value: any, valueType: number) => void
  addMetadata: (key: string, value: any, valueType: number) => void
  deleteMetadata: (key: string) => void
}

export const useFileStore = create<FileStore>((set, get) => ({
  fileInfo: null,
  isDirty: false,
  activeView: 'overview',
  selectedTensorIndex: null,

  setFileInfo: (info) =>
    set({ fileInfo: info, isDirty: false, selectedTensorIndex: null, activeView: 'overview' }),

  setActiveView: (view) => set({ activeView: view, selectedTensorIndex: null }),

  selectTensor: (index) => set({ selectedTensorIndex: index }),

  markDirty: () => set({ isDirty: true }),
  markClean: () => set({ isDirty: false }),

  updateMetadata: (key, value, valueType) => {
    const { fileInfo } = get()
    if (!fileInfo) return
    const metadata = fileInfo.metadata.map((kv) =>
      kv.key === key ? { ...kv, value, valueType } : kv
    )
    set({
      fileInfo: { ...fileInfo, metadata },
      isDirty: true
    })
  },

  addMetadata: (key, value, valueType) => {
    const { fileInfo } = get()
    if (!fileInfo) return
    set({
      fileInfo: {
        ...fileInfo,
        metadata: [...fileInfo.metadata, { key, value, valueType }],
        header: {
          ...fileInfo.header,
          metadataKvCount: fileInfo.metadata.length + 1
        }
      },
      isDirty: true
    })
  },

  deleteMetadata: (key) => {
    const { fileInfo } = get()
    if (!fileInfo) return
    const metadata = fileInfo.metadata.filter((kv) => kv.key !== key)
    set({
      fileInfo: {
        ...fileInfo,
        metadata,
        header: { ...fileInfo.header, metadataKvCount: metadata.length }
      },
      isDirty: true
    })
  }
}))
