import * as fs from 'fs'
import { GgufTensorInfo, TensorStats } from './types'
import { dequantize, canDequantize } from './dequantize'

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
  count: number
): Float32Array | null {
  if (!canDequantize(tensor.type)) return null

  const fd = fs.openSync(filePath, 'r')
  try {
    const absoluteOffset = dataStartOffset + tensor.offset + chunkOffset
    const readLen = Math.min(tensor.sizeBytes - chunkOffset, tensor.sizeBytes)
    const buf = Buffer.alloc(readLen)
    fs.readSync(fd, buf, 0, readLen, absoluteOffset)
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
    const chunkSize = 1024 * 1024 // 1MB chunks
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
