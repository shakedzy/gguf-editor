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
  displayName?: string
}

export type ActiveView = 'overview' | 'metadata' | 'tensors' | 'diagram'

// Byte-level edits per tensor: tensorIndex → Map<byteOffset, newValue>
export type ByteEditsMap = Record<number, [number, number][]>

interface FileStore {
  fileInfo: GgufFileInfo | null
  isDirty: boolean
  activeView: ActiveView
  selectedTensorIndex: number | null
  diagramFocusTensor: string | null // tensor name to scroll to in diagram
  byteEdits: ByteEditsMap // all pending byte edits across all tensors
  saveVersion: number // increments on save — signals components to flush caches

  setFileInfo: (info: GgufFileInfo | null) => void
  setActiveView: (view: ActiveView) => void
  selectTensor: (index: number | null) => void
  navigateToTensor: (index: number) => void
  showTensorInDiagram: (tensorName: string) => void
  clearDiagramFocus: () => void
  markDirty: () => void
  markClean: () => void
  setByteEditsForTensor: (tensorIndex: number, edits: Map<number, number>) => void
  updateMetadata: (key: string, value: any, valueType: number) => void
  addMetadata: (key: string, value: any, valueType: number) => void
  deleteMetadata: (key: string) => void
}

export const useFileStore = create<FileStore>((set, get) => ({
  fileInfo: null,
  isDirty: false,
  activeView: 'overview',
  selectedTensorIndex: null,
  diagramFocusTensor: null,
  byteEdits: {},
  saveVersion: 0,

  setFileInfo: (info) =>
    set({ fileInfo: info, isDirty: false, selectedTensorIndex: null, diagramFocusTensor: null, activeView: 'overview', byteEdits: {}, saveVersion: 0 }),

  setActiveView: (view) => set({ activeView: view, selectedTensorIndex: null }),

  selectTensor: (index) => set({ selectedTensorIndex: index }),

  navigateToTensor: (index) => set({ activeView: 'tensors', selectedTensorIndex: index }),

  showTensorInDiagram: (tensorName) => set({ activeView: 'diagram', diagramFocusTensor: tensorName }),

  clearDiagramFocus: () => set({ diagramFocusTensor: null }),

  markDirty: () => set({ isDirty: true }),
  markClean: () => set((s) => ({ isDirty: false, byteEdits: {}, saveVersion: s.saveVersion + 1 })),

  setByteEditsForTensor: (tensorIndex, edits) => {
    const { byteEdits } = get()
    const next = { ...byteEdits }
    if (edits.size === 0) {
      delete next[tensorIndex]
    } else {
      next[tensorIndex] = Array.from(edits.entries())
    }
    const hasAnyEdits = Object.keys(next).length > 0
    set({ byteEdits: next, isDirty: hasAnyEdits || get().isDirty })
  },

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
