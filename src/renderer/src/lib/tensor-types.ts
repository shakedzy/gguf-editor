export const GGML_TYPE_NAME: Record<number, string> = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  6: 'Q5_0',
  7: 'Q5_1',
  8: 'Q8_0',
  9: 'Q8_1',
  10: 'Q2_K',
  11: 'Q3_K',
  12: 'Q4_K',
  13: 'Q5_K',
  14: 'Q6_K',
  15: 'Q8_K',
  16: 'IQ2_XXS',
  17: 'IQ2_XS',
  18: 'IQ3_XXS',
  19: 'IQ1_S',
  20: 'IQ4_NL',
  21: 'IQ3_S',
  22: 'IQ2_S',
  23: 'IQ4_XS',
  24: 'IQ1_M',
  30: 'BF16'
}

export const METADATA_TYPE_NAME: Record<number, string> = {
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

export function getTypeBadgeColor(type: number): string {
  if (type <= 1) return 'bg-green-500/20 text-green-400' // F32, F16
  if (type <= 3) return 'bg-blue-500/20 text-blue-400' // Q4
  if (type <= 7) return 'bg-purple-500/20 text-purple-400' // Q5
  if (type <= 9) return 'bg-amber-500/20 text-amber-400' // Q8
  if (type <= 15) return 'bg-rose-500/20 text-rose-400' // K-quants
  return 'bg-gray-500/20 text-gray-400' // IQ / other
}
