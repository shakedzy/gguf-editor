export class BinaryReader {
  private buf: Buffer
  private pos: number

  constructor(buffer: Buffer) {
    this.buf = buffer
    this.pos = 0
  }

  get position(): number {
    return this.pos
  }

  set position(p: number) {
    this.pos = p
  }

  get remaining(): number {
    return this.buf.length - this.pos
  }

  readUint8(): number {
    const v = this.buf.readUInt8(this.pos)
    this.pos += 1
    return v
  }

  readInt8(): number {
    const v = this.buf.readInt8(this.pos)
    this.pos += 1
    return v
  }

  readUint16(): number {
    const v = this.buf.readUInt16LE(this.pos)
    this.pos += 2
    return v
  }

  readInt16(): number {
    const v = this.buf.readInt16LE(this.pos)
    this.pos += 2
    return v
  }

  readUint32(): number {
    const v = this.buf.readUInt32LE(this.pos)
    this.pos += 4
    return v
  }

  readInt32(): number {
    const v = this.buf.readInt32LE(this.pos)
    this.pos += 4
    return v
  }

  readUint64(): bigint {
    const v = this.buf.readBigUInt64LE(this.pos)
    this.pos += 8
    return v
  }

  readUint64AsNumber(): number {
    const v = this.readUint64()
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`uint64 value ${v} exceeds Number.MAX_SAFE_INTEGER`)
    }
    return Number(v)
  }

  readInt64(): bigint {
    const v = this.buf.readBigInt64LE(this.pos)
    this.pos += 8
    return v
  }

  readFloat32(): number {
    const v = this.buf.readFloatLE(this.pos)
    this.pos += 4
    return v
  }

  readFloat64(): number {
    const v = this.buf.readDoubleLE(this.pos)
    this.pos += 8
    return v
  }

  readFloat16(): number {
    const half = this.readUint16()
    return float16ToFloat32(half)
  }

  readBool(): boolean {
    return this.readUint8() !== 0
  }

  readString(): string {
    const len = this.readUint64AsNumber()
    const str = this.buf.toString('utf8', this.pos, this.pos + len)
    this.pos += len
    return str
  }

  readBytes(n: number): Buffer {
    const slice = this.buf.subarray(this.pos, this.pos + n)
    this.pos += n
    return slice
  }

  skip(n: number): void {
    this.pos += n
  }
}

export function float16ToFloat32(half: number): number {
  const sign = (half >> 15) & 0x1
  const exponent = (half >> 10) & 0x1f
  const fraction = half & 0x3ff

  if (exponent === 0) {
    if (fraction === 0) return sign ? -0 : 0
    // Subnormal
    const f = fraction / 1024
    const val = Math.pow(2, -14) * f
    return sign ? -val : val
  }

  if (exponent === 0x1f) {
    if (fraction === 0) return sign ? -Infinity : Infinity
    return NaN
  }

  const val = Math.pow(2, exponent - 15) * (1 + fraction / 1024)
  return sign ? -val : val
}
