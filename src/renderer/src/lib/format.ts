export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const k = 1000
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  const val = bytes / Math.pow(k, i)
  return `${val.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export function formatNumber(n: number): string {
  return n.toLocaleString()
}

export function formatHex(n: number, width = 8): string {
  return '0x' + n.toString(16).padStart(width, '0').toUpperCase()
}

export function formatShape(dims: number[]): string {
  return dims.join(' x ')
}

export function formatFloat(v: number, precision = 6): string {
  if (!isFinite(v)) return String(v)
  return v.toPrecision(precision)
}
