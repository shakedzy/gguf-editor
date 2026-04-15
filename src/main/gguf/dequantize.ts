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
    GgmlType.Q8_1
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
// dequant: w[i] = d * (nibble[i] - 8)
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
// dequant: w[i] = d * nibble[i] + m
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
// dequant: w[i] = d * (q5[i] - 16), q5 = low4 | (bit_from_qh << 4)
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

      const hBitLo = (qh >> i) & 1
      const hBitHi = (qh >> (i + 16)) & 1

      const q5Lo = lo | (hBitLo << 4)
      const q5Hi = hi | (hBitHi << 4)

      result[b * 32 + i] = d * (q5Lo - 16)
      result[b * 32 + i + 16] = d * (q5Hi - 16)
    }
  }

  return result
}

// Q5_1: block of 32 weights = 2B scale + 2B min + 4B high-bits + 16B low nibbles
// dequant: w[i] = d * q5[i] + m
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

      const hBitLo = (qh >> i) & 1
      const hBitHi = (qh >> (i + 16)) & 1

      const q5Lo = lo | (hBitLo << 4)
      const q5Hi = hi | (hBitHi << 4)

      result[b * 32 + i] = d * q5Lo + m
      result[b * 32 + i + 16] = d * q5Hi + m
    }
  }

  return result
}

// Q8_0: block of 32 weights = 2B scale + 32B int8 values
// dequant: w[i] = d * qs[i]
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
// dequant: w[i] = d * qs[i]  (sum is for dot-product optimization, not used in dequant)
function dequantizeQ8_1(buf: Buffer): Float32Array {
  const blockSize = 36
  const numBlocks = Math.floor(buf.length / blockSize)
  const result = new Float32Array(numBlocks * 32)

  for (let b = 0; b < numBlocks; b++) {
    const offset = b * blockSize
    const d = float16ToFloat32(buf.readUInt16LE(offset))
    // skip 2 bytes for sum at offset+2

    for (let i = 0; i < 32; i++) {
      result[b * 32 + i] = d * buf.readInt8(offset + 4 + i)
    }
  }

  return result
}
