import { GgmlType } from './types'
import { float16ToFloat32 } from './reader'

export function canDequantize(type: GgmlType): boolean {
  return [
    GgmlType.F32,
    GgmlType.F16,
    GgmlType.Q4_0,
    GgmlType.Q4_1,
    GgmlType.Q5_0,
    GgmlType.Q5_1,
    GgmlType.Q8_0,
    GgmlType.Q8_1,
    GgmlType.Q4_K,
    GgmlType.Q6_K
  ].includes(type)
}


export function dequantize(
  type: GgmlType,
  buf: Buffer
): Float32Array | null {
  switch (type) {
    case GgmlType.F32:
      return dequantizeF32(buf)
    case GgmlType.F16:
      return dequantizeF16(buf)
    case GgmlType.Q4_0:
      return dequantizeQ4_0(buf)
    case GgmlType.Q4_1:
      return dequantizeQ4_1(buf)
    case GgmlType.Q5_0:
      return dequantizeQ5_0(buf)
    case GgmlType.Q5_1:
      return dequantizeQ5_1(buf)
    case GgmlType.Q8_0:
      return dequantizeQ8_0(buf)
    case GgmlType.Q8_1:
      return dequantizeQ8_1(buf)
    case GgmlType.Q4_K:
      return dequantizeQ4_K(buf)
    case GgmlType.Q6_K:
      return dequantizeQ6_K(buf)
    default:
      return null
  }
}

function dequantizeF32(buf: Buffer): Float32Array {
  const count = Math.floor(buf.length / 4)
  const result = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    result[i] = buf.readFloatLE(i * 4)
  }
  return result
}

function dequantizeF16(buf: Buffer): Float32Array {
  const count = Math.floor(buf.length / 2)
  const result = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    result[i] = float16ToFloat32(buf.readUInt16LE(i * 2))
  }
  return result
}

// Q4_0: block of 32 weights = 2B float16 scale + 16B packed nibbles
function dequantizeQ4_0(buf: Buffer): Float32Array {
  const blockSize = 18
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * 32)

  for (let b = 0; b < numBlocks; b++) {
    const offset = b * blockSize
    const d = float16ToFloat32(buf.readUInt16LE(offset))

    for (let i = 0; i < 16; i++) {
      const byte = buf[offset + 2 + i]
      const lo = byte & 0x0f
      const hi = (byte >> 4) & 0x0f
      result[b * 32 + i] = d * (lo - 8)
      result[b * 32 + i + 16] = d * (hi - 8)
    }
  }
  return result
}

// Q4_1: block of 32 weights = 2B scale + 2B min + 16B packed nibbles
function dequantizeQ4_1(buf: Buffer): Float32Array {
  const blockSize = 20
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * 32)

  for (let b = 0; b < numBlocks; b++) {
    const offset = b * blockSize
    const d = float16ToFloat32(buf.readUInt16LE(offset))
    const m = float16ToFloat32(buf.readUInt16LE(offset + 2))

    for (let i = 0; i < 16; i++) {
      const byte = buf[offset + 4 + i]
      const lo = byte & 0x0f
      const hi = (byte >> 4) & 0x0f
      result[b * 32 + i] = d * lo + m
      result[b * 32 + i + 16] = d * hi + m
    }
  }
  return result
}

// Q5_0: block of 32 weights = 2B scale + 4B high-bits + 16B low nibbles
function dequantizeQ5_0(buf: Buffer): Float32Array {
  const blockSize = 22
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * 32)

  for (let b = 0; b < numBlocks; b++) {
    const offset = b * blockSize
    const d = float16ToFloat32(buf.readUInt16LE(offset))
    const qh = buf.readUInt32LE(offset + 2)

    for (let i = 0; i < 16; i++) {
      const byte = buf[offset + 6 + i]
      const lo = byte & 0x0f
      const hi = (byte >> 4) & 0x0f
      result[b * 32 + i] = d * ((lo | (((qh >> i) & 1) << 4)) - 16)
      result[b * 32 + i + 16] = d * ((hi | (((qh >> (i + 16)) & 1) << 4)) - 16)
    }
  }
  return result
}

// Q5_1: block of 32 weights = 2B scale + 2B min + 4B high-bits + 16B low nibbles
function dequantizeQ5_1(buf: Buffer): Float32Array {
  const blockSize = 24
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * 32)

  for (let b = 0; b < numBlocks; b++) {
    const offset = b * blockSize
    const d = float16ToFloat32(buf.readUInt16LE(offset))
    const m = float16ToFloat32(buf.readUInt16LE(offset + 2))
    const qh = buf.readUInt32LE(offset + 4)

    for (let i = 0; i < 16; i++) {
      const byte = buf[offset + 8 + i]
      const lo = byte & 0x0f
      const hi = (byte >> 4) & 0x0f
      result[b * 32 + i] = d * (lo | (((qh >> i) & 1) << 4)) + m
      result[b * 32 + i + 16] = d * (hi | (((qh >> (i + 16)) & 1) << 4)) + m
    }
  }
  return result
}

// Q8_0: block of 32 weights = 2B scale + 32B int8 values
function dequantizeQ8_0(buf: Buffer): Float32Array {
  const blockSize = 34
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * 32)

  for (let b = 0; b < numBlocks; b++) {
    const offset = b * blockSize
    const d = float16ToFloat32(buf.readUInt16LE(offset))
    for (let i = 0; i < 32; i++) {
      result[b * 32 + i] = d * buf.readInt8(offset + 2 + i)
    }
  }
  return result
}

// Q8_1: block of 32 weights = 2B scale + 2B sum + 32B int8 values
function dequantizeQ8_1(buf: Buffer): Float32Array {
  const blockSize = 36
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * 32)

  for (let b = 0; b < numBlocks; b++) {
    const offset = b * blockSize
    const d = float16ToFloat32(buf.readUInt16LE(offset))
    for (let i = 0; i < 32; i++) {
      result[b * 32 + i] = d * buf.readInt8(offset + 4 + i)
    }
  }
  return result
}

// ── K-Quant types ───────────────────────────────────────────────────

// Helper: extract 6-bit scale and min for Q4_K sub-block j
function getScaleMinK4(
  j: number,
  scales: Buffer,
  scalesOff: number
): { sc: number; m: number } {
  if (j < 4) {
    return {
      sc: scales[scalesOff + j] & 63,
      m: scales[scalesOff + j + 4] & 63
    }
  }
  return {
    sc: (scales[scalesOff + j + 4] & 0xf) | ((scales[scalesOff + j - 4] >> 6) << 4),
    m: (scales[scalesOff + j + 4] >> 4) | ((scales[scalesOff + j] >> 6) << 4)
  }
}

// Q4_K: 144 bytes = 256 weights
// Layout: d(f16 2B) + dmin(f16 2B) + scales(12B) + qs(128B)
// 8 sub-blocks of 32 weights, processed in pairs of 64
function dequantizeQ4_K(buf: Buffer): Float32Array {
  const QK_K = 256
  const blockSize = 144
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * QK_K)

  for (let b = 0; b < numBlocks; b++) {
    const off = b * blockSize
    const d = float16ToFloat32(buf.readUInt16LE(off))
    const dmin = float16ToFloat32(buf.readUInt16LE(off + 2))
    const scalesOff = off + 4
    const qsOff = off + 16

    let outIdx = b * QK_K
    let qIdx = 0
    let is = 0

    for (let j = 0; j < QK_K; j += 64) {
      const { sc: sc1, m: m1 } = getScaleMinK4(is, buf, scalesOff)
      const d1 = d * sc1
      const dm1 = dmin * m1

      const { sc: sc2, m: m2 } = getScaleMinK4(is + 1, buf, scalesOff)
      const d2 = d * sc2
      const dm2 = dmin * m2

      for (let l = 0; l < 32; l++) {
        const q = buf[qsOff + qIdx + l]
        result[outIdx + l] = d1 * (q & 0xf) - dm1
        result[outIdx + l + 32] = d2 * (q >> 4) - dm2
      }

      outIdx += 64
      qIdx += 32
      is += 2
    }
  }
  return result
}

// Q6_K: 210 bytes = 256 weights
// Layout: ql(128B) + qh(64B) + scales(int8 x16, 16B) + d(f16 2B)
// 16 sub-blocks of 16 weights, processed in groups of 128
function dequantizeQ6_K(buf: Buffer): Float32Array {
  const QK_K = 256
  const blockSize = 210
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * QK_K)

  for (let b = 0; b < numBlocks; b++) {
    const off = b * blockSize
    const qlBase = off           // 128 bytes: lower 4 bits
    const qhBase = off + 128     // 64 bytes: upper 2 bits
    const scBase = off + 192     // 16 bytes: int8 scales
    const d = float16ToFloat32(buf.readUInt16LE(off + 208))

    let outIdx = b * QK_K
    let qlOff = 0
    let qhOff = 0
    let scOff = 0

    // Process in two halves of 128 weights each
    for (let n = 0; n < QK_K; n += 128) {
      for (let l = 0; l < 32; l++) {
        const ql0 = buf[qlBase + qlOff + l]
        const ql1 = buf[qlBase + qlOff + l + 32]
        const qhVal = buf[qhBase + qhOff + l]

        const q1 = ((ql0 & 0xf) | (((qhVal >> 0) & 3) << 4)) - 32
        const q2 = ((ql1 & 0xf) | (((qhVal >> 2) & 3) << 4)) - 32
        const q3 = ((ql0 >> 4) | (((qhVal >> 4) & 3) << 4)) - 32
        const q4 = ((ql1 >> 4) | (((qhVal >> 6) & 3) << 4)) - 32

        result[outIdx + l + 0] = d * buf.readInt8(scBase + scOff + 0) * q1
        result[outIdx + l + 32] = d * buf.readInt8(scBase + scOff + 2) * q2
        result[outIdx + l + 64] = d * buf.readInt8(scBase + scOff + 4) * q3
        result[outIdx + l + 96] = d * buf.readInt8(scBase + scOff + 6) * q4
      }

      qlOff += 64
      qhOff += 32
      scOff += 8
      outIdx += 128
    }
  }
  return result
}
