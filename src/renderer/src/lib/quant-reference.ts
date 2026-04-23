import mdSource from '../../../../docs/QUANT_TYPES.md?raw'

// Split the reference doc by `### TYPE` (or `### TYPE · TYPE · TYPE`) headings.
// Each type name maps to its section's markdown body (heading included).

const sections = new Map<string, string>()

function parseSections(source: string): void {
  const lines = source.split('\n')
  let currentTypes: string[] = []
  let currentLines: string[] = []

  const flush = () => {
    if (currentTypes.length === 0) return
    const body = currentLines.join('\n').trim()
    if (body) {
      for (const t of currentTypes) {
        sections.set(t, body)
      }
    }
    currentTypes = []
    currentLines = []
  }

  for (const line of lines) {
    const h3 = /^###\s+(.+?)\s*$/.exec(line)
    if (h3) {
      flush()
      currentTypes = h3[1]
        .split('·')
        .map((s) => s.trim())
        .filter(Boolean)
      currentLines = [line]
      continue
    }
    if (/^##\s+/.test(line)) {
      flush()
      continue
    }
    if (currentTypes.length > 0) {
      currentLines.push(line)
    }
  }
  flush()
}

parseSections(mdSource)

export function getReferenceSection(typeName: string): string | null {
  return sections.get(typeName) ?? null
}
