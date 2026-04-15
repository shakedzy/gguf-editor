import * as fs from 'fs'
import { BinaryReader } from './reader'
import {
  GgufHeader,
  GgufMetadataKV,
  GgufMetadataValue,
  GgufMetadataValueType,
  GgufMetadataArrayValue,
  GgufTensorInfo,
  GgufFileInfo,
  GgmlType
} from './types'
import { GGUF_MAGIC, GGUF_DEFAULT_ALIGNMENT, computeTensorSizeBytes } from './constants'

function readMetadataValue(
  reader: BinaryReader,
  valueType: GgufMetadataValueType
): GgufMetadataValue {
  switch (valueType) {
    case GgufMetadataValueType.UINT8:
      return reader.readUint8()
    case GgufMetadataValueType.INT8:
      return reader.readInt8()
    case GgufMetadataValueType.UINT16:
      return reader.readUint16()
    case GgufMetadataValueType.INT16:
      return reader.readInt16()
    case GgufMetadataValueType.UINT32:
      return reader.readUint32()
    case GgufMetadataValueType.INT32:
      return reader.readInt32()
    case GgufMetadataValueType.FLOAT32:
      return reader.readFloat32()
    case GgufMetadataValueType.BOOL:
      return reader.readBool()
    case GgufMetadataValueType.STRING:
      return reader.readString()
    case GgufMetadataValueType.ARRAY: {
      const arrType = reader.readUint32() as GgufMetadataValueType
      const count = reader.readUint64AsNumber()
      const values: (number | bigint | boolean | string)[] = []
      for (let i = 0; i < count; i++) {
        values.push(
          readMetadataValue(reader, arrType) as
            | number
            | bigint
            | boolean
            | string
        )
      }
      return { type: arrType, values } as GgufMetadataArrayValue
    }
    case GgufMetadataValueType.UINT64:
      return reader.readUint64()
    case GgufMetadataValueType.INT64:
      return reader.readInt64()
    case GgufMetadataValueType.FLOAT64:
      return reader.readFloat64()
    default:
      throw new Error(`Unknown metadata value type: ${valueType}`)
  }
}

export async function parseGgufFile(filePath: string): Promise<GgufFileInfo> {
  const fd = fs.openSync(filePath, 'r')
  try {
    const stat = fs.fstatSync(fd)
    const fileSize = stat.size

    // Read a generous initial chunk for header + metadata + tensor info.
    // Metadata can be large (tokenizer data), so start with 64MB and extend if needed.
    const initialReadSize = Math.min(fileSize, 64 * 1024 * 1024)
    let buffer = Buffer.alloc(initialReadSize)
    fs.readSync(fd, buffer, 0, initialReadSize, 0)

    const reader = new BinaryReader(buffer)

    // Parse header
    const magic = reader.readUint32()
    if (magic !== GGUF_MAGIC) {
      throw new Error(
        `Invalid GGUF magic: 0x${magic.toString(16).padStart(8, '0')} (expected 0x${GGUF_MAGIC.toString(16).padStart(8, '0')})`
      )
    }

    const version = reader.readUint32()
    if (version < 2 || version > 3) {
      throw new Error(`Unsupported GGUF version: ${version}`)
    }

    const tensorCount = reader.readUint64AsNumber()
    const metadataKvCount = reader.readUint64AsNumber()

    const header: GgufHeader = { magic, version, tensorCount, metadataKvCount }

    // Helper to ensure we have enough data read from the file
    const ensureData = (needed: number): void => {
      if (reader.position + needed > buffer.length) {
        const newSize = Math.min(
          fileSize,
          Math.max(buffer.length * 2, reader.position + needed + 1024 * 1024)
        )
        if (newSize <= buffer.length) {
          throw new Error(
            'GGUF header/metadata section exceeds file size'
          )
        }
        const newBuffer = Buffer.alloc(newSize)
        buffer.copy(newBuffer)
        fs.readSync(fd, newBuffer, buffer.length, newSize - buffer.length, buffer.length)
        const pos = reader.position
        ;(reader as any).buf = newBuffer
        reader.position = pos
        buffer = newBuffer
      }
    }

    // Parse metadata KV pairs
    const metadata: GgufMetadataKV[] = []
    for (let i = 0; i < metadataKvCount; i++) {
      ensureData(1024) // ensure at least 1KB ahead
      const key = reader.readString()
      const valueType = reader.readUint32() as GgufMetadataValueType
      const value = readMetadataValue(reader, valueType)
      metadata.push({ key, valueType, value })
    }

    // Determine alignment from metadata
    let alignment = GGUF_DEFAULT_ALIGNMENT
    const alignmentKv = metadata.find((kv) => kv.key === 'general.alignment')
    if (alignmentKv && typeof alignmentKv.value === 'number') {
      alignment = alignmentKv.value
    }

    // Parse tensor info
    const tensors: GgufTensorInfo[] = []
    for (let i = 0; i < tensorCount; i++) {
      ensureData(1024)
      const name = reader.readString()
      const nDims = reader.readUint32()
      const dims: number[] = []
      for (let d = 0; d < nDims; d++) {
        dims.push(reader.readUint64AsNumber())
      }
      const type = reader.readUint32() as GgmlType
      const offset = reader.readUint64AsNumber()
      const sizeBytes = computeTensorSizeBytes(dims, type)
      tensors.push({ name, nDims, dims, type, offset, sizeBytes })
    }

    // Compute data start offset (aligned)
    const headerEndPos = reader.position
    const dataStartOffset =
      headerEndPos + ((alignment - (headerEndPos % alignment)) % alignment)

    return {
      header,
      metadata,
      tensors,
      alignment,
      dataStartOffset,
      fileSize,
      filePath
    }
  } finally {
    fs.closeSync(fd)
  }
}
