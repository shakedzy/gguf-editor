export interface GgufHeader {
  magic: number
  version: number
  tensorCount: number
  metadataKvCount: number
}

export enum GgufMetadataValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12
}

export interface GgufMetadataKV {
  key: string
  valueType: GgufMetadataValueType
  value: GgufMetadataValue
}

export type GgufMetadataValue =
  | number
  | bigint
  | boolean
  | string
  | GgufMetadataArrayValue

export interface GgufMetadataArrayValue {
  type: GgufMetadataValueType
  values: (number | bigint | boolean | string)[]
}

export enum GgmlType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  IQ1_M = 24,
  BF16 = 30
}

export interface GgufTensorInfo {
  name: string
  nDims: number
  dims: number[]
  type: GgmlType
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

export interface TensorStats {
  min: number
  max: number
  mean: number
  stddev: number
  count: number
}
