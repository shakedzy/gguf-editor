// Shared block-layout helpers: used by HexEditor (coloring, dividers)
// and DequantizedView (mapping byte edits back to affected weights).

export interface BlockDef {
  blockBytes: number
  weightsPerBlock: number
  segments: { start: number; len: number; role: string }[]
  // Sub-block boundaries within the block (byte offsets relative to block start).
  // Used to draw thin dividers showing logical sub-block groupings.
  subBlockBoundaries?: number[]
}

export const BLOCK_DEFS: Record<number, BlockDef> = {
  0:  { blockBytes: 4, weightsPerBlock: 1, segments: [{ start: 0, len: 4, role: 'value' }] },
  1:  { blockBytes: 2, weightsPerBlock: 1, segments: [{ start: 0, len: 2, role: 'value' }] },
  30: { blockBytes: 2, weightsPerBlock: 1, segments: [{ start: 0, len: 2, role: 'value' }] },

  2: {
    blockBytes: 18, weightsPerBlock: 32,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 16, role: 'quants' }]
  },
  3: {
    blockBytes: 20, weightsPerBlock: 32,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 2, role: 'min' }, { start: 4, len: 16, role: 'quants' }]
  },
  6: {
    blockBytes: 22, weightsPerBlock: 32,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 4, role: 'highbits' }, { start: 6, len: 16, role: 'quants' }]
  },
  7: {
    blockBytes: 24, weightsPerBlock: 32,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 2, role: 'min' }, { start: 4, len: 4, role: 'highbits' }, { start: 8, len: 16, role: 'quants' }]
  },
  8: {
    blockBytes: 34, weightsPerBlock: 32,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 32, role: 'quants' }]
  },
  9: {
    blockBytes: 36, weightsPerBlock: 32,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 2, role: 'sum' }, { start: 4, len: 32, role: 'quants' }]
  },

  10: {
    blockBytes: 84, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 16, role: 'subscale' },
      { start: 16, len: 64, role: 'quants' },
      { start: 80, len: 2, role: 'scale' },
      { start: 82, len: 2, role: 'min' }
    ]
  },
  11: {
    blockBytes: 110, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 32, role: 'highbits' },
      { start: 32, len: 64, role: 'quants' },
      { start: 96, len: 12, role: 'subscale' },
      { start: 108, len: 2, role: 'scale' }
    ]
  },
  12: {
    blockBytes: 144, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 2, role: 'scale' },
      { start: 2, len: 2, role: 'min' },
      { start: 4, len: 4, role: 'subscale' },
      { start: 8, len: 4, role: 'submin' },
      { start: 12, len: 4, role: 'submixed' },
      { start: 16, len: 128, role: 'quants' }
    ],
    // 8 sub-blocks of 32 weights. Packed scales encode one (scale, min) per
    // sub-block; each qs byte encodes one nibble in each of two sub-blocks.
    // Dividers inside packed scales (per byte) + every 32 bytes of qs.
    subBlockBoundaries: [5, 6, 7, 9, 10, 11, 13, 14, 15, 48, 80, 112]
  },
  13: {
    blockBytes: 176, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 2, role: 'scale' },
      { start: 2, len: 2, role: 'min' },
      { start: 4, len: 4, role: 'subscale' },
      { start: 8, len: 4, role: 'submin' },
      { start: 12, len: 4, role: 'submixed' },
      { start: 16, len: 32, role: 'highbits' },
      { start: 48, len: 128, role: 'quants' }
    ],
    subBlockBoundaries: [5, 6, 7, 9, 10, 11, 13, 14, 15, 80, 112, 144]
  },
  14: {
    blockBytes: 210, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 128, role: 'quants' },
      { start: 128, len: 64, role: 'highbits' },
      { start: 192, len: 16, role: 'subscale' },
      { start: 208, len: 2, role: 'scale' }
    ],
    // Q6_K: 16 sub-blocks of 16 weights. scales[] is 1 byte per sub-block.
    subBlockBoundaries: [193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207]
  },
  15: {
    blockBytes: 292, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 4, role: 'scale' },
      { start: 4, len: 256, role: 'quants' },
      { start: 260, len: 32, role: 'sum' }
    ]
  },

  16: {
    blockBytes: 66, weightsPerBlock: 256,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 64, role: 'quants' }]
  },
  17: {
    blockBytes: 74, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 2, role: 'scale' },
      { start: 2, len: 64, role: 'quants' },
      { start: 66, len: 8, role: 'subscale' }
    ]
  },
  18: {
    blockBytes: 98, weightsPerBlock: 256,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 96, role: 'quants' }]
  },
  19: {
    blockBytes: 50, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 2, role: 'scale' },
      { start: 2, len: 32, role: 'quants' },
      { start: 34, len: 16, role: 'highbits' }
    ]
  },
  20: {
    blockBytes: 18, weightsPerBlock: 32,
    segments: [{ start: 0, len: 2, role: 'scale' }, { start: 2, len: 16, role: 'quants' }]
  },
  21: {
    blockBytes: 110, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 2, role: 'scale' },
      { start: 2, len: 64, role: 'quants' },
      { start: 66, len: 8, role: 'highbits' },
      { start: 74, len: 32, role: 'signs' },
      { start: 106, len: 4, role: 'subscale' }
    ]
  },
  22: {
    blockBytes: 82, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 2, role: 'scale' },
      { start: 2, len: 64, role: 'quants' },
      { start: 66, len: 8, role: 'highbits' },
      { start: 74, len: 8, role: 'subscale' }
    ]
  },
  23: {
    blockBytes: 136, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 2, role: 'scale' },
      { start: 2, len: 6, role: 'subscale' },
      { start: 8, len: 128, role: 'quants' }
    ]
  },
  24: {
    blockBytes: 56, weightsPerBlock: 256,
    segments: [
      { start: 0, len: 32, role: 'quants' },
      { start: 32, len: 16, role: 'highbits' },
      { start: 48, len: 8, role: 'subscale' }
    ]
  }
}

function range(start: number, end: number): number[] {
  const out: number[] = []
  for (let i = start; i < end; i++) out.push(i)
  return out
}

// Given a byte offset within a block, return the weight indices (within the
// same block) whose dequantized value changes when that byte changes.
// Returns null for types where the mapping is unknown — caller should treat
// as "whole block".
export function affectedWeightsInBlock(
  type: number,
  byteInBlock: number
): number[] | null {
  switch (type) {
    case 0: // F32: 4 bytes per weight
      return [Math.floor(byteInBlock / 4)]
    case 1: // F16
    case 30: // BF16
      return [Math.floor(byteInBlock / 2)]

    case 2: { // Q4_0: d(0-1) + qs(2-17)
      if (byteInBlock < 2) return range(0, 32)
      const i = byteInBlock - 2
      return [i, i + 16]
    }
    case 3: { // Q4_1: d(0-1) + m(2-3) + qs(4-19)
      if (byteInBlock < 4) return range(0, 32)
      const i = byteInBlock - 4
      return [i, i + 16]
    }
    case 6: { // Q5_0: d(0-1) + qh(2-5) + qs(6-21)
      if (byteInBlock < 2) return range(0, 32)
      if (byteInBlock < 6) {
        const bitStart = (byteInBlock - 2) * 8 // qh bit i = high bit of weight i
        return range(bitStart, bitStart + 8)
      }
      const i = byteInBlock - 6
      return [i, i + 16]
    }
    case 7: { // Q5_1: d(0-1) + m(2-3) + qh(4-7) + qs(8-23)
      if (byteInBlock < 4) return range(0, 32)
      if (byteInBlock < 8) {
        const bitStart = (byteInBlock - 4) * 8
        return range(bitStart, bitStart + 8)
      }
      const i = byteInBlock - 8
      return [i, i + 16]
    }
    case 8: { // Q8_0: d(0-1) + qs(2-33)
      if (byteInBlock < 2) return range(0, 32)
      return [byteInBlock - 2]
    }
    case 9: { // Q8_1: d(0-1) + sum(2-3) + qs(4-35)
      if (byteInBlock < 4) return range(0, 32)
      return [byteInBlock - 4]
    }

    case 12: { // Q4_K: d(0-1) + dmin(2-3) + packed scales(4-15) + qs(16-143)
      if (byteInBlock < 4) return range(0, 256) // d or dmin
      if (byteInBlock < 8) {
        // byte 4+j: low 6 bits = scale[j]; high 2 bits = upper bits of scale[j+4]
        const j = byteInBlock - 4
        return [...range(j * 32, (j + 1) * 32), ...range((j + 4) * 32, (j + 5) * 32)]
      }
      if (byteInBlock < 12) {
        // byte 8+j: low 6 bits = min[j]; high 2 bits = upper bits of min[j+4]
        const j = byteInBlock - 8
        return [...range(j * 32, (j + 1) * 32), ...range((j + 4) * 32, (j + 5) * 32)]
      }
      if (byteInBlock < 16) {
        // byte 12+j: low 4 bits = scale[j+4] low; high 4 bits = min[j+4] low
        const j = byteInBlock - 12
        return range((j + 4) * 32, (j + 5) * 32)
      }
      // qs[q] for q in 0..127: low nibble and high nibble land in two
      // different sub-blocks (a pair), but at the same position `l` within
      // the sub-block. See dequantizeQ4_K for the exact layout.
      const q = byteInBlock - 16
      const group = Math.floor(q / 32) // 0..3
      const l = q % 32
      const sbLow = 2 * group
      const sbHigh = sbLow + 1
      return [sbLow * 32 + l, sbHigh * 32 + l]
    }

    case 14: { // Q6_K: ql(0-127) + qh(128-191) + scales(192-207) + d(208-209)
      if (byteInBlock >= 208) return range(0, 256) // d affects everything
      if (byteInBlock >= 192) {
        // scales[j] applies to 16 weights; the ggml dequant loop picks
        // sc[is + 0,2,4,6] with is = l/16, so scales[0..7] serve the first
        // 128 weights and scales[8..15] serve the second 128.
        // Within one half, scales[0,2,4,6] each apply to weights
        // [l, l+32, l+64, l+96] with l in [0,16) -> sub-block indexed by is.
        // Simpler affected-set: each sc byte affects exactly 16 contiguous
        // weights at stride 32 in the proper half.
        const j = byteInBlock - 192 // 0..15
        const half = Math.floor(j / 8) // 0 or 1
        const k = j % 8 // 0..7
        const is = k % 2 // 0 or 1 (l/16 bit)
        const stripe = Math.floor(k / 2) // 0..3 -> *32 offset
        const halfStart = half * 128
        const stripeStart = halfStart + stripe * 32 + is * 16
        return range(stripeStart, stripeStart + 16)
      }
      if (byteInBlock >= 128) {
        // qh byte at index qh (0..63): within half = qh/32, l = qh%32,
        // contributes 2 bits each to weights [l, l+32, l+64, l+96].
        const qh = byteInBlock - 128
        const half = Math.floor(qh / 32)
        const l = qh % 32
        const h = half * 128
        return [h + l, h + l + 32, h + l + 64, h + l + 96]
      }
      // ql byte at index ql (0..127): within half = ql/64, rem = ql%64.
      // rem in [0,32) -> ql0: low nibble -> weight l, high nibble -> weight l+64
      // rem in [32,64) -> ql1: low nibble -> weight l+32, high nibble -> weight l+96
      const ql = byteInBlock
      const half = Math.floor(ql / 64)
      const rem = ql % 64
      const l = rem % 32
      const h = half * 128
      if (rem < 32) return [h + l, h + l + 64]
      return [h + l + 32, h + l + 96]
    }

    default:
      return null
  }
}
