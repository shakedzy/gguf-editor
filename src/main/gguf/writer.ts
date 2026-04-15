import * as fs from 'fs'
import {
  GgufFileInfo,
  GgufMetadataKV,
  GgufMetadataValueType,
  GgufMetadataValue,
  GgufMetadataArrayValue
} from './types'
import { GGUF_MAGIC } from './constants'

class BinaryWriter {
  private chunks: Buffer[] = []
  private _size = 0

  get size(): number {
    return this._size
  }

  writeUint8(v: number): void {
    const buf = Buffer.alloc(1)
    buf.writeUInt8(v)
    this.chunks.push(buf)
    this._size += 1
  }

  writeInt8(v: number): void {
    const buf = Buffer.alloc(1)
    buf.writeInt8(v)
    this.chunks.push(buf)
    this._size += 1
  }

  writeUint16(v: number): void {
    const buf = Buffer.alloc(2)
    buf.writeUInt16LE(v)
    this.chunks.push(buf)
    this._size += 2
  }

  writeInt16(v: number): void {
    const buf = Buffer.alloc(2)
    buf.writeInt16LE(v)
    this.chunks.push(buf)
    this._size += 2
  }

  writeUint32(v: number): void {
    const buf = Buffer.alloc(4)
    buf.writeUInt32LE(v)
    this.chunks.push(buf)
    this._size += 4
  }

  writeInt32(v: number): void {
    const buf = Buffer.alloc(4)
    buf.writeInt32LE(v)
    this.chunks.push(buf)
    this._size += 4
  }

  writeUint64(v: number | bigint): void {
    const buf = Buffer.alloc(8)
    buf.writeBigUInt64LE(BigInt(v))
    this.chunks.push(buf)
    this._size += 8
  }

  writeInt64(v: number | bigint): void {
    const buf = Buffer.alloc(8)
    buf.writeBigInt64LE(BigInt(v))
    this.chunks.push(buf)
    this._size += 8
  }

  writeFloat32(v: number): void {
    const buf = Buffer.alloc(4)
    buf.writeFloatLE(v)
    this.chunks.push(buf)
    this._size += 4
  }

  writeFloat64(v: number): void {
    const buf = Buffer.alloc(8)
    buf.writeDoubleLE(v)
    this.chunks.push(buf)
    this._size += 8
  }

  writeBool(v: boolean): void {
    this.writeUint8(v ? 1 : 0)
  }

  writeString(s: string): void {
    const strBuf = Buffer.from(s, 'utf8')
    this.writeUint64(strBuf.length)
    this.chunks.push(strBuf)
    this._size += strBuf.length
  }

  writePadding(alignment: number): void {
    const pad = (alignment - (this._size % alignment)) % alignment
    if (pad > 0) {
      const buf = Buffer.alloc(pad, 0)
      this.chunks.push(buf)
      this._size += pad
    }
  }

  toBuffer(): Buffer {
    return Buffer.concat(this.chunks)
  }
}

function writeMetadataValue(
  writer: BinaryWriter,
  valueType: GgufMetadataValueType,
  value: GgufMetadataValue
): void {
  switch (valueType) {
    case GgufMetadataValueType.UINT8:
      writer.writeUint8(value as number)
      break
    case GgufMetadataValueType.INT8:
      writer.writeInt8(value as number)
      break
    case GgufMetadataValueType.UINT16:
      writer.writeUint16(value as number)
      break
    case GgufMetadataValueType.INT16:
      writer.writeInt16(value as number)
      break
    case GgufMetadataValueType.UINT32:
      writer.writeUint32(value as number)
      break
    case GgufMetadataValueType.INT32:
      writer.writeInt32(value as number)
      break
    case GgufMetadataValueType.FLOAT32:
      writer.writeFloat32(value as number)
      break
    case GgufMetadataValueType.BOOL:
      writer.writeBool(value as boolean)
      break
    case GgufMetadataValueType.STRING:
      writer.writeString(value as string)
      break
    case GgufMetadataValueType.ARRAY: {
      const arr = value as GgufMetadataArrayValue
      writer.writeUint32(arr.type)
      writer.writeUint64(arr.values.length)
      for (const v of arr.values) {
        writeMetadataValue(writer, arr.type, v)
      }
      break
    }
    case GgufMetadataValueType.UINT64:
      writer.writeUint64(value as bigint)
      break
    case GgufMetadataValueType.INT64:
      writer.writeInt64(value as bigint)
      break
    case GgufMetadataValueType.FLOAT64:
      writer.writeFloat64(value as number)
      break
  }
}

export async function writeGgufFile(
  fileInfo: GgufFileInfo,
  outputPath: string
): Promise<void> {
  const writer = new BinaryWriter()

  // Header
  writer.writeUint32(GGUF_MAGIC)
  writer.writeUint32(fileInfo.header.version)
  writer.writeUint64(fileInfo.header.tensorCount)
  writer.writeUint64(fileInfo.metadata.length)

  // Metadata KV pairs
  for (const kv of fileInfo.metadata) {
    writer.writeString(kv.key)
    writer.writeUint32(kv.valueType)
    writeMetadataValue(writer, kv.valueType, kv.value)
  }

  // Tensor info
  for (const t of fileInfo.tensors) {
    writer.writeString(t.name)
    writer.writeUint32(t.nDims)
    for (let d = 0; d < t.nDims; d++) {
      writer.writeUint64(t.dims[d])
    }
    writer.writeUint32(t.type)
    writer.writeUint64(t.offset)
  }

  // Pad to alignment
  writer.writePadding(fileInfo.alignment)

  const headerBuffer = writer.toBuffer()

  // Write header, then stream tensor data from source file
  const outFd = fs.openSync(outputPath, 'w')
  try {
    fs.writeSync(outFd, headerBuffer, 0, headerBuffer.length, 0)

    // Copy tensor data from source
    const srcFd = fs.openSync(fileInfo.filePath, 'r')
    try {
      const copyChunkSize = 64 * 1024 // 64KB
      const tensorDataSize = fileInfo.fileSize - fileInfo.dataStartOffset
      let copied = 0
      const copyBuf = Buffer.alloc(copyChunkSize)

      while (copied < tensorDataSize) {
        const toRead = Math.min(copyChunkSize, tensorDataSize - copied)
        const bytesRead = fs.readSync(
          srcFd,
          copyBuf,
          0,
          toRead,
          fileInfo.dataStartOffset + copied
        )
        if (bytesRead === 0) break
        fs.writeSync(outFd, copyBuf, 0, bytesRead, headerBuffer.length + copied)
        copied += bytesRead
      }
    } finally {
      fs.closeSync(srcFd)
    }
  } finally {
    fs.closeSync(outFd)
  }
}
