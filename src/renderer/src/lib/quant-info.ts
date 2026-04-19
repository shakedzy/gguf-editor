import { GGML_TYPE_NAME } from './tensor-types'

export interface QuantExplanation {
  summary: string
  blockLayout: string
  formula: string
  bytesPerFloat: string
}

const EXPLANATIONS: Record<number, QuantExplanation> = {
  // ── Unquantized ────────────────────────────────────────────────────
  0: {
    summary: 'Each weight is stored as a standard 32-bit IEEE 754 float. No quantization — every 4 bytes maps directly to one float value.',
    blockLayout: '4 bytes per weight: [float32]',
    formula: 'value = read_float32(bytes)',
    bytesPerFloat: '4.0 bytes/weight'
  },
  1: {
    summary: 'Each weight is stored as a 16-bit IEEE 754 half-precision float. Every 2 bytes maps directly to one float value.',
    blockLayout: '2 bytes per weight: [float16]',
    formula: 'value = float16_to_float32(bytes)',
    bytesPerFloat: '2.0 bytes/weight'
  },
  30: {
    summary: 'Each weight is stored as a 16-bit bfloat16 (brain floating point). Like float32 but truncated to 16 bits — same exponent range, less precision.',
    blockLayout: '2 bytes per weight: [bfloat16]',
    formula: 'value = bfloat16_to_float32(bytes)',
    bytesPerFloat: '2.0 bytes/weight'
  },

  // ── Simple quants (32 weights/block) ───────────────────────────────
  2: {
    summary: 'Symmetric 4-bit quantization. Blocks of 32 weights. Each block: 2-byte float16 scale (d), then 16 bytes of packed 4-bit nibbles (2 weights per byte).',
    blockLayout: '18 bytes = 32 weights: [d: f16 (2B)] [nibbles (16B)]',
    formula: 'value = d × (nibble − 8)',
    bytesPerFloat: '0.5625 bytes/weight (4.5 bits)'
  },
  3: {
    summary: 'Asymmetric 4-bit quantization. Like Q4_0 but with an additional minimum offset per block.',
    blockLayout: '20 bytes = 32 weights: [d: f16 (2B)] [m: f16 (2B)] [nibbles (16B)]',
    formula: 'value = d × nibble + m',
    bytesPerFloat: '0.625 bytes/weight (5 bits)'
  },
  6: {
    summary: 'Symmetric 5-bit quantization. Blocks of 32. The 5th bit for each weight comes from a 4-byte high-bit mask.',
    blockLayout: '22 bytes = 32 weights: [d: f16 (2B)] [high bits (4B)] [low nibbles (16B)]',
    formula: 'value = d × (q5 − 16), where q5 = low4 | (high_bit << 4)',
    bytesPerFloat: '0.6875 bytes/weight (5.5 bits)'
  },
  7: {
    summary: 'Asymmetric 5-bit quantization. Like Q5_0 but with an additional minimum offset per block.',
    blockLayout: '24 bytes = 32 weights: [d: f16 (2B)] [m: f16 (2B)] [high bits (4B)] [low nibbles (16B)]',
    formula: 'value = d × q5 + m',
    bytesPerFloat: '0.75 bytes/weight (6 bits)'
  },
  8: {
    summary: 'Symmetric 8-bit quantization. Blocks of 32. The first 2 bytes are the scale factor (d), NOT a weight — bytes 02-33 are the 32 signed int8 quantized weights.',
    blockLayout: '34 bytes = 32 weights: [d: f16 (2B)] [qs: int8 × 32 (32B)]',
    formula: 'value = d × qs[i]',
    bytesPerFloat: '1.0625 bytes/weight (8.5 bits)'
  },
  9: {
    summary: 'Asymmetric 8-bit quantization. Like Q8_0 but with an additional 2-byte sum field (used for optimized dot products).',
    blockLayout: '36 bytes = 32 weights: [d: f16 (2B)] [sum: f16 (2B)] [qs: int8 × 32 (32B)]',
    formula: 'value = d × qs[i]',
    bytesPerFloat: '1.125 bytes/weight (9 bits)'
  },

  // ── K-quants (256 weights/block) ───────────────────────────────────
  10: {
    summary: 'K-quant 2-bit quantization. Blocks of 256 weights with 16 sub-blocks. Sub-block scales and mins are packed in the first 16 bytes. Super-block scale (d) and min (dmin) are at the end.',
    blockLayout: '84 bytes = 256 weights: [sub-scales (16B)] [2-bit quants (64B)] [d: f16 (2B)] [dmin: f16 (2B)]',
    formula: 'value = d × sub_scale × q2 − dmin × sub_min',
    bytesPerFloat: '0.328 bytes/weight (2.6 bits)'
  },
  11: {
    summary: 'K-quant 3-bit quantization. Blocks of 256 weights. High-bit mask (32B) provides the 3rd bit for each weight. Sub-block scales (12B) and super-block scale (d) are at the end.',
    blockLayout: '110 bytes = 256 weights: [high-bit mask (32B)] [2-bit quants (64B)] [sub-scales (12B)] [d: f16 (2B)]',
    formula: 'value = d × sub_scale × (q3 − 4), where q3 = low2 | (high_bit << 2)',
    bytesPerFloat: '0.43 bytes/weight (3.4 bits)'
  },
  12: {
    summary: 'K-quant 4-bit quantization. Blocks of 256 weights with 8 sub-blocks of 32 each. Super-block scale (d) and min (dmin) at the start, followed by packed 6-bit sub-block scales/mins, then 128 bytes of 4-bit nibbles.',
    blockLayout: '144 bytes = 256 weights: [d (2B)] [dmin (2B)] [sub-scales (4B)] [sub-mins (4B)] [overflow (4B)] [4-bit quants (128B)]',
    formula: 'value = d × sub_scale × nibble − dmin × sub_min',
    bytesPerFloat: '0.5625 bytes/weight (4.5 bits)'
  },
  13: {
    summary: 'K-quant 5-bit quantization. Blocks of 256 weights with 8 sub-blocks of 32 each. Like Q4_K but with an additional 32-byte high-bit field providing the 5th bit for each weight.',
    blockLayout: '176 bytes = 256 weights: [d (2B)] [dmin (2B)] [sub-scales (12B)] [high bits (32B)] [4-bit quants (128B)]',
    formula: 'value = d × sub_scale × q5 − dmin × sub_min, where q5 = low4 | (high_bit << 4)',
    bytesPerFloat: '0.6875 bytes/weight (5.5 bits)'
  },
  14: {
    summary: 'K-quant 6-bit quantization. Blocks of 256 weights. Lower 4 bits (128B) and upper 2 bits (64B) are stored separately. 16 int8 sub-block scales and a super-block scale (d) are at the end.',
    blockLayout: '210 bytes = 256 weights: [low 4-bit quants (128B)] [high 2-bit quants (64B)] [sub-scales (16B)] [d: f16 (2B)]',
    formula: 'value = d × sub_scale × (q6 − 32), where q6 = low4 | (high2 << 4)',
    bytesPerFloat: '0.82 bytes/weight (6.5 bits)'
  },
  15: {
    summary: 'K-quant 8-bit quantization. Blocks of 256 weights. Scale is float32 (4 bytes, not float16). Includes 32 bytes of int16 block sums for optimized dot products.',
    blockLayout: '292 bytes = 256 weights: [d: f32 (4B)] [qs: int8 × 256 (256B)] [block sums: int16 × 16 (32B)]',
    formula: 'value = d × qs[i]',
    bytesPerFloat: '1.14 bytes/weight (9.1 bits)'
  },

  // ── IQ types (importance-matrix quants) ────────────────────────────
  16: {
    summary: 'Importance-matrix 2-bit quantization (extra-extra-small). Blocks of 256 weights. Uses hardcoded codebook lookup tables — the 64 bytes of quants are indices into the IQ2_XXS grid, not raw integers.',
    blockLayout: '66 bytes = 256 weights: [d: f16 (2B)] [codebook indices (64B)]',
    formula: 'value = d × codebook_lookup(index) — codebook is hardcoded in ggml',
    bytesPerFloat: '0.258 bytes/weight (2.06 bits)'
  },
  17: {
    summary: 'Importance-matrix 2-bit quantization (extra-small). Like IQ2_XXS but with an 8-byte sub-block scales field for finer per-sub-block adjustment.',
    blockLayout: '74 bytes = 256 weights: [d: f16 (2B)] [codebook indices (64B)] [sub-scales (8B)]',
    formula: 'value = d × sub_scale × codebook_lookup(index)',
    bytesPerFloat: '0.289 bytes/weight (2.31 bits)'
  },
  18: {
    summary: 'Importance-matrix 3-bit quantization (extra-extra-small). Blocks of 256 weights. The 96 bytes of quants encode 3-bit codebook indices with sign information interleaved.',
    blockLayout: '98 bytes = 256 weights: [d: f16 (2B)] [packed 3-bit indices + signs (96B)]',
    formula: 'value = d × sign × codebook_lookup(index)',
    bytesPerFloat: '0.383 bytes/weight (3.06 bits)'
  },
  19: {
    summary: 'Importance-matrix 1-bit quantization (small). Blocks of 256 weights. Extremely compressed — uses codebook indices (32B) plus high-bit field (16B) to reconstruct values.',
    blockLayout: '50 bytes = 256 weights: [d: f16 (2B)] [codebook indices (32B)] [high bits (16B)]',
    formula: 'value = d × codebook_lookup(index, high_bits)',
    bytesPerFloat: '0.195 bytes/weight (1.56 bits)'
  },
  20: {
    summary: 'Importance-matrix 4-bit quantization (non-linear). Same block layout as Q4_0 (18 bytes = 32 weights), but nibbles index into a non-linear lookup table instead of using linear (nibble − 8) mapping.',
    blockLayout: '18 bytes = 32 weights: [d: f16 (2B)] [nibbles (16B)]',
    formula: 'value = d × NL_LUT[nibble] — NL_LUT is a hardcoded 16-entry non-linear table',
    bytesPerFloat: '0.5625 bytes/weight (4.5 bits)'
  },
  21: {
    summary: 'Importance-matrix 3-bit quantization (small). Blocks of 256 weights. Has separate fields for quant indices, high bits, sign bits, and sub-block scales.',
    blockLayout: '110 bytes = 256 weights: [d: f16 (2B)] [3-bit indices (64B)] [high bits (8B)] [sign bits (32B)] [sub-scales (4B)]',
    formula: 'value = d × sub_scale × sign × codebook_lookup(index, high_bits)',
    bytesPerFloat: '0.43 bytes/weight (3.44 bits)'
  },
  22: {
    summary: 'Importance-matrix 2-bit quantization (small). Blocks of 256 weights with codebook indices, high bits, and sub-block scales.',
    blockLayout: '82 bytes = 256 weights: [d: f16 (2B)] [codebook indices (64B)] [high bits (8B)] [sub-scales (8B)]',
    formula: 'value = d × sub_scale × codebook_lookup(index, high_bits)',
    bytesPerFloat: '0.32 bytes/weight (2.56 bits)'
  },
  23: {
    summary: 'Importance-matrix 4-bit quantization (extra-small). Blocks of 256 weights. Like IQ4_NL but with sub-block scales for 256-weight blocks instead of 32-weight blocks.',
    blockLayout: '136 bytes = 256 weights: [d: f16 (2B)] [sub-scales (6B)] [nibbles (128B)]',
    formula: 'value = d × sub_scale × NL_LUT[nibble]',
    bytesPerFloat: '0.531 bytes/weight (4.25 bits)'
  },
  24: {
    summary: 'Importance-matrix 1-bit quantization (medium). Blocks of 256 weights. No separate super-block scale field — scale is packed into the 8-byte scales field along with sub-block info.',
    blockLayout: '56 bytes = 256 weights: [codebook indices (32B)] [high bits (16B)] [packed scales (8B)]',
    formula: 'value = packed_scale × codebook_lookup(index, high_bits)',
    bytesPerFloat: '0.219 bytes/weight (1.75 bits)'
  }
}

export function getQuantExplanation(type: number): QuantExplanation | null {
  return EXPLANATIONS[type] ?? null
}

export function getHexExplanation(type: number): string {
  const typeName = GGML_TYPE_NAME[type] ?? `type ${type}`
  const info = EXPLANATIONS[type]

  if (!info) {
    return `Raw bytes of ${typeName}-quantized tensor data. Block structure unknown — use Float Values tab for decoded weights.`
  }

  // Unquantized types: bytes map directly to values
  if (type === 0 || type === 1 || type === 30) {
    return `${info.summary} Formula: ${info.formula}`
  }

  return `${typeName}: ${info.blockLayout}. Formula: ${info.formula}`
}

export function getFloatExplanation(type: number): string {
  const typeName = GGML_TYPE_NAME[type] ?? `type ${type}`
  const info = EXPLANATIONS[type]

  if (!info) {
    return `Dequantized weight values from ${typeName} tensor.`
  }

  return `These are the actual weight values after dequantizing from ${typeName}. ${info.formula}. Each value here was reconstructed from the compressed block data shown in the Hex View.`
}

// ── Structured formula for rendering ────────────────────────────────

export interface FormulaPart {
  text: string
  role?: string   // matches ROLE_COLOR keys: 'scale', 'min', 'subscale', 'submin', 'quants', 'highbits', 'signs', 'value'
  sub?: string    // subscript text
  sup?: string    // superscript text
}

export interface FormulaVariable {
  parts: FormulaPart[]  // the variable notation
  desc: string          // what it is
}

export interface StructuredFormula {
  equation: FormulaPart[]
  variables: FormulaVariable[]
  note?: string
}

const FORMULAS: Record<number, StructuredFormula> = {
  // F32
  0: {
    equation: [
      { text: 'value' }, { text: ' = ' }, { text: 'float32', role: 'value' }, { text: '(bytes)' }
    ],
    variables: [
      { parts: [{ text: 'bytes' }], desc: '4 bytes, IEEE 754' }
    ]
  },
  // F16
  1: {
    equation: [
      { text: 'value' }, { text: ' = ' }, { text: 'float16', role: 'value' }, { text: '(bytes)' }
    ],
    variables: [
      { parts: [{ text: 'bytes' }], desc: '2 bytes, IEEE 754 half-precision' }
    ]
  },
  // BF16
  30: {
    equation: [
      { text: 'value' }, { text: ' = ' }, { text: 'bfloat16', role: 'value' }, { text: '(bytes)' }
    ],
    variables: [
      { parts: [{ text: 'bytes' }], desc: '2 bytes, same exponent as float32, 8-bit mantissa' }
    ]
  },
  // Q4_0
  2: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 (' },
      { text: 'q', sub: 'i', role: 'quants' }, { text: ' \u2212 8)' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'q', sub: 'i', role: 'quants' }], desc: '4-bit nibble, 0\u201315' }
    ]
  },
  // Q4_1
  3: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'q', sub: 'i', role: 'quants' }, { text: ' + ' },
      { text: 'm', role: 'min' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'm', role: 'min' }], desc: 'block minimum (float16)' },
      { parts: [{ text: 'q', sub: 'i', role: 'quants' }], desc: '4-bit nibble, 0\u201315' }
    ]
  },
  // Q5_0
  6: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 (' },
      { text: 'q5', sub: 'i', role: 'quants' }, { text: ' \u2212 16)' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'q5', sub: 'i', role: 'quants' }], desc: 'low4 | (high_bit << 4), 0\u201331' },
      { parts: [{ text: 'high_bit', role: 'highbits' }], desc: '5th bit from 4-byte mask' }
    ]
  },
  // Q5_1
  7: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'q5', sub: 'i', role: 'quants' }, { text: ' + ' },
      { text: 'm', role: 'min' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'm', role: 'min' }], desc: 'block minimum (float16)' },
      { parts: [{ text: 'q5', sub: 'i', role: 'quants' }], desc: 'low4 | (high_bit << 4), 0\u201331' }
    ]
  },
  // Q8_0
  8: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'qs', sub: 'i', role: 'quants' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'qs', sub: 'i', role: 'quants' }], desc: 'signed int8, \u2212128\u2026127' }
    ]
  },
  // Q8_1
  9: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'qs', sub: 'i', role: 'quants' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'qs', sub: 'i', role: 'quants' }], desc: 'signed int8, \u2212128\u2026127' },
      { parts: [{ text: 'sum', role: 'sum' }], desc: 'block sum (float16), used for dot products' }
    ]
  },
  // Q2_K
  10: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 ' },
      { text: 'q', sub: 'i', role: 'quants' }, { text: ' \u2212 ' },
      { text: 'd', role: 'scale', sub: 'min' }, { text: ' \u00d7 ' },
      { text: 'm', sub: 'j', role: 'subscale' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }, { text: ', ' }, { text: 'd', role: 'scale', sub: 'min' }], desc: 'super-block scale & min (float16, at end of block)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }, { text: ', ' }, { text: 'm', sub: 'j', role: 'subscale' }], desc: 'sub-block scale & min (4-bit, packed in first 16B)' },
      { parts: [{ text: 'q', sub: 'i', role: 'quants' }], desc: '2-bit value, 0\u20133' }
    ]
  },
  // Q3_K
  11: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 (' },
      { text: 'q3', sub: 'i', role: 'quants' }, { text: ' \u2212 4)' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'super-block scale (float16, at end of block)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'sub-block scale (6-bit packed in 12B)' },
      { parts: [{ text: 'q3', sub: 'i', role: 'quants' }], desc: 'low2 | (hmask_bit << 2), 0\u20137' },
      { parts: [{ text: 'hmask', role: 'highbits' }], desc: '3rd bit from 32-byte mask' }
    ]
  },
  // Q4_K
  12: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 ' },
      { text: 'q', sub: 'i', role: 'quants' }, { text: ' \u2212 ' },
      { text: 'd', role: 'min', sub: 'min' }, { text: ' \u00d7 ' },
      { text: 'm', sub: 'j', role: 'submin' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'super-block scale (float16)' },
      { parts: [{ text: 'd', role: 'min', sub: 'min' }], desc: 'super-block min (float16)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'sub-block scale (6-bit)' },
      { parts: [{ text: 'm', sub: 'j', role: 'submin' }], desc: 'sub-block min (6-bit)' },
      { parts: [{ text: 'q', sub: 'i', role: 'quants' }], desc: '4-bit nibble, 0\u201315' }
    ],
    note: '8 sub-blocks of 32 weights each'
  },
  // Q5_K
  13: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 ' },
      { text: 'q5', sub: 'i', role: 'quants' }, { text: ' \u2212 ' },
      { text: 'd', role: 'min', sub: 'min' }, { text: ' \u00d7 ' },
      { text: 'm', sub: 'j', role: 'submin' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'super-block scale (float16)' },
      { parts: [{ text: 'd', role: 'min', sub: 'min' }], desc: 'super-block min (float16)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'sub-block scale (6-bit)' },
      { parts: [{ text: 'm', sub: 'j', role: 'submin' }], desc: 'sub-block min (6-bit)' },
      { parts: [{ text: 'q5', sub: 'i', role: 'quants' }], desc: 'low4 | (high_bit << 4), 0\u201331' },
      { parts: [{ text: 'high_bit', role: 'highbits' }], desc: '5th bit from 32-byte mask' }
    ],
    note: '8 sub-blocks of 32 weights each'
  },
  // Q6_K
  14: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 (' },
      { text: 'q6', sub: 'i', role: 'quants' }, { text: ' \u2212 32)' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'super-block scale (float16, at end of block)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'sub-block scale (int8)' },
      { parts: [{ text: 'q6', sub: 'i', role: 'quants' }], desc: 'low4 | (high2 << 4), 0\u201363' },
      { parts: [{ text: 'high2', role: 'highbits' }], desc: 'upper 2 bits from 64-byte field' }
    ],
    note: '16 sub-blocks of 16 weights each'
  },
  // Q8_K
  15: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'qs', sub: 'i', role: 'quants' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float32 \u2014 4 bytes, not float16)' },
      { parts: [{ text: 'qs', sub: 'i', role: 'quants' }], desc: 'signed int8, \u2212128\u2026127' },
      { parts: [{ text: 'bsums', role: 'sum' }], desc: 'int16 block sums, for dot products' }
    ]
  },
  // IQ2_XXS
  16: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'LUT', role: 'quants' }, { text: '[' },
      { text: 'idx', sub: 'i', role: 'quants' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'idx', sub: 'i', role: 'quants' }], desc: 'codebook index from packed data' },
      { parts: [{ text: 'LUT' }], desc: 'hardcoded IQ2_XXS grid in ggml' }
    ]
  },
  // IQ2_XS
  17: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 ' },
      { text: 'LUT', role: 'quants' }, { text: '[' },
      { text: 'idx', sub: 'i', role: 'quants' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'sub-block scale (from 8B scales)' },
      { parts: [{ text: 'idx', sub: 'i', role: 'quants' }], desc: 'codebook index' }
    ]
  },
  // IQ3_XXS
  18: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 sign \u00d7 ' },
      { text: 'LUT', role: 'quants' }, { text: '[' },
      { text: 'idx', sub: 'i', role: 'quants' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'idx', sub: 'i', role: 'quants' }], desc: '3-bit codebook index + sign, interleaved in 96B' }
    ]
  },
  // IQ1_S
  19: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'LUT', role: 'quants' }, { text: '[' },
      { text: 'idx', sub: 'i', role: 'quants' }, { text: ', ' },
      { text: 'qh', sub: 'i', role: 'highbits' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'idx', sub: 'i', role: 'quants' }], desc: 'codebook index (from 32B)' },
      { parts: [{ text: 'qh', sub: 'i', role: 'highbits' }], desc: 'high bits (from 16B)' }
    ]
  },
  // IQ4_NL
  20: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'NL', role: 'quants' }, { text: '[' },
      { text: 'q', sub: 'i', role: 'quants' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'q', sub: 'i', role: 'quants' }], desc: '4-bit nibble, 0\u201315' },
      { parts: [{ text: 'NL' }], desc: 'non-linear 16-entry lookup table (hardcoded in ggml)' }
    ]
  },
  // IQ3_S
  21: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 ' },
      { text: 'sign', sub: 'i', role: 'signs' }, { text: ' \u00d7 ' },
      { text: 'LUT', role: 'quants' }, { text: '[' },
      { text: 'idx', sub: 'i', role: 'quants' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'sub-block scale (from 4B)' },
      { parts: [{ text: 'sign', sub: 'i', role: 'signs' }], desc: '\u00b11 from 32-byte sign field' },
      { parts: [{ text: 'idx', sub: 'i', role: 'quants' }], desc: '3-bit codebook index (64B + 8B high bits)' }
    ]
  },
  // IQ2_S
  22: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 ' },
      { text: 'LUT', role: 'quants' }, { text: '[' },
      { text: 'idx', sub: 'i', role: 'quants' }, { text: ', ' },
      { text: 'qh', sub: 'i', role: 'highbits' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'sub-block scale (from 8B)' },
      { parts: [{ text: 'idx', sub: 'i', role: 'quants' }], desc: 'codebook index (from 64B)' },
      { parts: [{ text: 'qh', sub: 'i', role: 'highbits' }], desc: 'high bits (from 8B)' }
    ]
  },
  // IQ4_XS
  23: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'd', role: 'scale' }, { text: ' \u00d7 ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 ' },
      { text: 'NL', role: 'quants' }, { text: '[' },
      { text: 'q', sub: 'i', role: 'quants' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'd', role: 'scale' }], desc: 'block scale (float16)' },
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'sub-block scale (6B packed)' },
      { parts: [{ text: 'q', sub: 'i', role: 'quants' }], desc: '4-bit nibble, 0\u201315' },
      { parts: [{ text: 'NL' }], desc: 'non-linear 16-entry lookup table' }
    ]
  },
  // IQ1_M
  24: {
    equation: [
      { text: 'w', sub: 'i' }, { text: ' = ' },
      { text: 'sc', sub: 'j', role: 'subscale' }, { text: ' \u00d7 ' },
      { text: 'LUT', role: 'quants' }, { text: '[' },
      { text: 'idx', sub: 'i', role: 'quants' }, { text: ', ' },
      { text: 'qh', sub: 'i', role: 'highbits' }, { text: ']' }
    ],
    variables: [
      { parts: [{ text: 'sc', sub: 'j', role: 'subscale' }], desc: 'packed scale (8B, includes super-block d)' },
      { parts: [{ text: 'idx', sub: 'i', role: 'quants' }], desc: 'codebook index (from 32B)' },
      { parts: [{ text: 'qh', sub: 'i', role: 'highbits' }], desc: 'high bits (from 16B)' }
    ],
    note: 'No separate d field \u2014 scale is packed into the scales array'
  }
}

export function getStructuredFormula(type: number): StructuredFormula | null {
  return FORMULAS[type] ?? null
}
