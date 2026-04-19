import { GgmlType } from './types'

export const GGUF_MAGIC = 0x46554747 // "GGUF" in little-endian (bytes: 47 47 55 46)
export const GGUF_HEADER_SIZE = 24
export const GGUF_DEFAULT_ALIGNMENT = 32

export interface QuantBlockInfo {
  blockSize: number
  bytesPerBlock: number
}

export const QUANT_BLOCK_INFO: Partial<Record<GgmlType, QuantBlockInfo>> = {
  [GgmlType.F32]: { blockSize: 1, bytesPerBlock: 4 },
  [GgmlType.F16]: { blockSize: 1, bytesPerBlock: 2 },
  [GgmlType.BF16]: { blockSize: 1, bytesPerBlock: 2 },
  [GgmlType.Q4_0]: { blockSize: 32, bytesPerBlock: 18 },
  [GgmlType.Q4_1]: { blockSize: 32, bytesPerBlock: 20 },
  [GgmlType.Q5_0]: { blockSize: 32, bytesPerBlock: 22 },
  [GgmlType.Q5_1]: { blockSize: 32, bytesPerBlock: 24 },
  [GgmlType.Q8_0]: { blockSize: 32, bytesPerBlock: 34 },
  [GgmlType.Q8_1]: { blockSize: 32, bytesPerBlock: 36 },
  [GgmlType.Q2_K]: { blockSize: 256, bytesPerBlock: 84 },
  [GgmlType.Q3_K]: { blockSize: 256, bytesPerBlock: 110 },
  [GgmlType.Q4_K]: { blockSize: 256, bytesPerBlock: 144 },
  [GgmlType.Q5_K]: { blockSize: 256, bytesPerBlock: 176 },
  [GgmlType.Q6_K]: { blockSize: 256, bytesPerBlock: 210 },
  [GgmlType.Q8_K]: { blockSize: 256, bytesPerBlock: 292 },
  [GgmlType.IQ2_XXS]: { blockSize: 256, bytesPerBlock: 66 },
  [GgmlType.IQ2_XS]: { blockSize: 256, bytesPerBlock: 74 },
  [GgmlType.IQ3_XXS]: { blockSize: 256, bytesPerBlock: 98 },
  [GgmlType.IQ1_S]: { blockSize: 256, bytesPerBlock: 50 },
  [GgmlType.IQ4_NL]: { blockSize: 32, bytesPerBlock: 18 },
  [GgmlType.IQ3_S]: { blockSize: 256, bytesPerBlock: 110 },
  [GgmlType.IQ2_S]: { blockSize: 256, bytesPerBlock: 82 },
  [GgmlType.IQ4_XS]: { blockSize: 256, bytesPerBlock: 136 },
  [GgmlType.IQ1_M]: { blockSize: 256, bytesPerBlock: 56 }
}

export const GGML_TYPE_NAME: Record<number, string> = {
  [GgmlType.F32]: 'F32',
  [GgmlType.F16]: 'F16',
  [GgmlType.Q4_0]: 'Q4_0',
  [GgmlType.Q4_1]: 'Q4_1',
  [GgmlType.Q5_0]: 'Q5_0',
  [GgmlType.Q5_1]: 'Q5_1',
  [GgmlType.Q8_0]: 'Q8_0',
  [GgmlType.Q8_1]: 'Q8_1',
  [GgmlType.Q2_K]: 'Q2_K',
  [GgmlType.Q3_K]: 'Q3_K',
  [GgmlType.Q4_K]: 'Q4_K',
  [GgmlType.Q5_K]: 'Q5_K',
  [GgmlType.Q6_K]: 'Q6_K',
  [GgmlType.Q8_K]: 'Q8_K',
  [GgmlType.IQ2_XXS]: 'IQ2_XXS',
  [GgmlType.IQ2_XS]: 'IQ2_XS',
  [GgmlType.IQ3_XXS]: 'IQ3_XXS',
  [GgmlType.IQ1_S]: 'IQ1_S',
  [GgmlType.IQ4_NL]: 'IQ4_NL',
  [GgmlType.IQ3_S]: 'IQ3_S',
  [GgmlType.IQ2_S]: 'IQ2_S',
  [GgmlType.IQ4_XS]: 'IQ4_XS',
  [GgmlType.IQ1_M]: 'IQ1_M',
  [GgmlType.BF16]: 'BF16'
}

export const METADATA_VALUE_TYPE_NAME: Record<number, string> = {
  0: 'UINT8',
  1: 'INT8',
  2: 'UINT16',
  3: 'INT16',
  4: 'UINT32',
  5: 'INT32',
  6: 'FLOAT32',
  7: 'BOOL',
  8: 'STRING',
  9: 'ARRAY',
  10: 'UINT64',
  11: 'INT64',
  12: 'FLOAT64'
}

export function computeTensorSizeBytes(
  dims: number[],
  type: GgmlType
): number {
  const info = QUANT_BLOCK_INFO[type]
  if (!info) return 0

  const elementCount = dims.reduce((a, b) => a * b, 1)
  const numBlocks = Math.ceil(elementCount / info.blockSize)
  return numBlocks * info.bytesPerBlock
}
