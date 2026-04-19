import * as fs from 'fs'
import * as path from 'path'
import * as https from 'https'
import { app } from 'electron'

const SOURCE_URL =
  'https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp'

export interface ArchRegistryData {
  // tensor name pattern → LLM_TENSOR_* role (e.g. "blk.%d.attn_q" → "LLM_TENSOR_ATTN_Q")
  tensorRoles: Record<string, string>
  // list of known architecture names (e.g. "llama", "gemma4", "clip")
  knownArchitectures: string[]
  // when this was fetched
  fetchedAt: string
}

export interface ArchRegistryResult extends ArchRegistryData {
  source: 'live' | 'cached' | 'offline'
}

function getCachePath(): string {
  return path.join(app.getPath('userData'), 'arch-registry.json')
}

function readCache(): ArchRegistryData | null {
  try {
    const data = fs.readFileSync(getCachePath(), 'utf8')
    return JSON.parse(data)
  } catch {
    return null
  }
}

function writeCache(data: ArchRegistryData): void {
  try {
    fs.writeFileSync(getCachePath(), JSON.stringify(data, null, 2))
  } catch {
    // Ignore write errors
  }
}

function fetchUrl(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const request = https.get(url, (res) => {
      if (res.statusCode !== 200) {
        reject(new Error(`HTTP ${res.statusCode}`))
        return
      }
      let data = ''
      res.on('data', (chunk: string) => (data += chunk))
      res.on('end', () => resolve(data))
    })
    request.on('error', reject)
    request.setTimeout(10000, () => {
      request.destroy()
      reject(new Error('Timeout'))
    })
  })
}

function parseCppSource(src: string): ArchRegistryData {
  const tensorRoles: Record<string, string> = {}
  const knownArchitectures: string[] = []

  // Parse LLM_TENSOR_NAMES map
  const tensorSection = src.match(/LLM_TENSOR_NAMES\s*=\s*\{([\s\S]*?)\};/)
  if (tensorSection) {
    const regex = /\{\s*LLM_TENSOR_(\w+),\s*"([^"]+)"\s*\}/g
    let match
    while ((match = regex.exec(tensorSection[1])) !== null) {
      tensorRoles[match[2]] = 'LLM_TENSOR_' + match[1]
    }
  }

  // Parse LLM_ARCH_NAMES map
  const archSection = src.match(/LLM_ARCH_NAMES\s*=\s*\{([\s\S]*?)\};/)
  if (archSection) {
    const regex = /\{\s*LLM_ARCH_\w+,\s*"([^"]+)"\s*\}/g
    let match
    while ((match = regex.exec(archSection[1])) !== null) {
      if (match[1] !== '(unknown)') knownArchitectures.push(match[1])
    }
  }

  return {
    tensorRoles,
    knownArchitectures,
    fetchedAt: new Date().toISOString()
  }
}

export async function getArchRegistry(): Promise<ArchRegistryResult> {
  const cached = readCache()
  if (cached && Object.keys(cached.tensorRoles).length > 0) {
    return { ...cached, source: 'cached' }
  }

  // No cache — must fetch
  return refreshArchRegistry()
}

export async function refreshArchRegistry(): Promise<ArchRegistryResult> {
  try {
    const src = await fetchUrl(SOURCE_URL)
    const data = parseCppSource(src)

    if (Object.keys(data.tensorRoles).length > 50) {
      writeCache(data)
      return { ...data, source: 'live' }
    }
  } catch {
    // Fetch failed
  }

  // Fall back to cache if refresh failed
  const cached = readCache()
  if (cached && Object.keys(cached.tensorRoles).length > 0) {
    return { ...cached, source: 'cached' }
  }

  return {
    tensorRoles: {},
    knownArchitectures: [],
    fetchedAt: '',
    source: 'offline'
  }
}
