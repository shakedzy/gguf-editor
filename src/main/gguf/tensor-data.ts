import * as fs from 'fs'
import { GgufTensorInfo, TensorStats, GgmlType } from './types'
import { dequantize, canDequantize } from './dequantize'
import { GGML_TYPE_NAME, QUANT_BLOCK_INFO } from './constants'

export function readTensorChunk(
  filePath: string,
  dataStartOffset: number,
  tensor: GgufTensorInfo,
  chunkOffset: number,
  chunkLength: number
): Buffer {
  const fd = fs.openSync(filePath, 'r')
  try {
    const absoluteOffset = dataStartOffset + tensor.offset + chunkOffset
    const readLen = Math.min(chunkLength, tensor.sizeBytes - chunkOffset)
    const buf = Buffer.alloc(readLen)
    fs.readSync(fd, buf, 0, readLen, absoluteOffset)
    return buf
  } finally {
    fs.closeSync(fd)
  }
}

export function dequantizeTensorChunk(
  filePath: string,
  dataStartOffset: number,
  tensor: GgufTensorInfo,
  chunkOffset: number,
  count: number,
  edits?: [number, number][]
): Float32Array | null {
  if (!canDequantize(tensor.type)) return null

  const fd = fs.openSync(filePath, 'r')
  try {
    const absoluteOffset = dataStartOffset + tensor.offset + chunkOffset
    const readLen = Math.min(tensor.sizeBytes - chunkOffset, tensor.sizeBytes)
    const buf = Buffer.alloc(readLen)
    fs.readSync(fd, buf, 0, readLen, absoluteOffset)

    // Apply byte-level edits before dequantizing
    if (edits) {
      for (const [offset, value] of edits) {
        const bufIdx = offset - chunkOffset
        if (bufIdx >= 0 && bufIdx < readLen) {
          buf[bufIdx] = value
        }
      }
    }

    const floats = dequantize(tensor.type, buf)
    if (!floats) return null
    return count < floats.length ? floats.slice(0, count) : floats
  } finally {
    fs.closeSync(fd)
  }
}

export function computeTensorStats(
  filePath: string,
  dataStartOffset: number,
  tensor: GgufTensorInfo
): TensorStats | null {
  if (!canDequantize(tensor.type)) return null

  const fd = fs.openSync(filePath, 'r')
  try {
    const absoluteOffset = dataStartOffset + tensor.offset
    const blockInfo = QUANT_BLOCK_INFO[tensor.type]
    const blockBytes = blockInfo?.bytesPerBlock ?? 1
    const chunkSize = Math.floor((1024 * 1024) / blockBytes) * blockBytes || (1024 * 1024)
    let offset = 0
    let min = Infinity
    let max = -Infinity
    let count = 0
    let mean = 0
    let m2 = 0 // For Welford's algorithm

    while (offset < tensor.sizeBytes) {
      const readLen = Math.min(chunkSize, tensor.sizeBytes - offset)
      const buf = Buffer.alloc(readLen)
      fs.readSync(fd, buf, 0, readLen, absoluteOffset + offset)

      const floats = dequantize(tensor.type, buf)
      if (!floats) return null

      for (let i = 0; i < floats.length; i++) {
        const v = floats[i]
        if (v < min) min = v
        if (v > max) max = v
        count++
        const delta = v - mean
        mean += delta / count
        const delta2 = v - mean
        m2 += delta * delta2
      }

      offset += readLen
    }

    const variance = count > 1 ? m2 / (count - 1) : 0
    return {
      min,
      max,
      mean,
      stddev: Math.sqrt(variance),
      count
    }
  } finally {
    fs.closeSync(fd)
  }
}

// ── Export tensor ────────────────────────────────────────────────────

/**
 * Export raw tensor bytes to a file (exact on-disk representation).
 */
export function exportTensorRaw(
  filePath: string,
  dataStartOffset: number,
  tensor: GgufTensorInfo,
  outputPath: string
): void {
  const srcFd = fs.openSync(filePath, 'r')
  const dstFd = fs.openSync(outputPath, 'w')
  try {
    const chunkSize = 64 * 1024
    let offset = 0
    const buf = Buffer.alloc(chunkSize)
    while (offset < tensor.sizeBytes) {
      const readLen = Math.min(chunkSize, tensor.sizeBytes - offset)
      fs.readSync(srcFd, buf, 0, readLen, dataStartOffset + tensor.offset + offset)
      fs.writeSync(dstFd, buf, 0, readLen)
      offset += readLen
    }
  } finally {
    fs.closeSync(srcFd)
    fs.closeSync(dstFd)
  }
}

/**
 * Export tensor as .npy (NumPy format) with dequantized float32 values.
 * NumPy v1.0 format: magic + header + raw float32 data.
 */
export function exportTensorNumpy(
  filePath: string,
  dataStartOffset: number,
  tensor: GgufTensorInfo,
  outputPath: string
): boolean {
  if (!canDequantize(tensor.type)) return false

  const totalElements = tensor.dims.reduce((a, b) => a * b, 1)

  // Build numpy header
  // Shape in numpy order (reversed from GGUF which stores row-major reversed)
  const shapeStr = '(' + tensor.dims.join(', ') + (tensor.dims.length === 1 ? ',' : '') + ')'
  const descrStr = "'<f4'" // little-endian float32
  const headerDict = `{'descr': ${descrStr}, 'fortran_order': False, 'shape': ${shapeStr}, }`

  // Pad header to 64-byte alignment (numpy requirement)
  const magicLen = 10 // \x93NUMPY\x01\x00 + 2-byte header len
  const totalHeaderLen = magicLen + headerDict.length + 1 // +1 for \n
  const padded = Math.ceil(totalHeaderLen / 64) * 64
  const padding = padded - totalHeaderLen
  const headerStr = headerDict + ' '.repeat(padding) + '\n'

  const headerBuf = Buffer.alloc(padded)
  // Magic: \x93NUMPY
  headerBuf[0] = 0x93
  headerBuf.write('NUMPY', 1, 'ascii')
  // Version 1.0
  headerBuf[6] = 1
  headerBuf[7] = 0
  // Header length (little-endian uint16)
  headerBuf.writeUInt16LE(headerStr.length, 8)
  // Header string
  headerBuf.write(headerStr, 10, 'ascii')

  const dstFd = fs.openSync(outputPath, 'w')
  const srcFd = fs.openSync(filePath, 'r')
  try {
    // Write numpy header
    fs.writeSync(dstFd, headerBuf, 0, headerBuf.length)

    // Stream dequantized data
    const blockInfo = QUANT_BLOCK_INFO[tensor.type]
    const blockBytes = blockInfo?.bytesPerBlock ?? 1
    const chunkSize = Math.floor((1024 * 1024) / blockBytes) * blockBytes || (1024 * 1024)
    let offset = 0
    while (offset < tensor.sizeBytes) {
      const readLen = Math.min(chunkSize, tensor.sizeBytes - offset)
      const buf = Buffer.alloc(readLen)
      fs.readSync(srcFd, buf, 0, readLen, dataStartOffset + tensor.offset + offset)

      const floats = dequantize(tensor.type, buf)
      if (!floats) return false

      const outBuf = Buffer.from(floats.buffer, floats.byteOffset, floats.byteLength)
      fs.writeSync(dstFd, outBuf, 0, outBuf.length)
      offset += readLen
    }

    return true
  } finally {
    fs.closeSync(srcFd)
    fs.closeSync(dstFd)
  }
}
