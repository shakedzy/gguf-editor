import { ReactNode } from 'react'

// Minimal GFM-ish renderer tailored to docs/QUANT_TYPES.md.
// Handles: #/##/### headings, paragraphs, ``` code blocks, | tables,
// - lists, ---, inline **bold**, *italic*, `code`, [label](url).

export function renderMarkdown(source: string): ReactNode {
  const lines = source.split('\n')
  const blocks: ReactNode[] = []
  let i = 0
  let key = 0
  const nextKey = () => key++

  while (i < lines.length) {
    const line = lines[i]

    if (/^---+\s*$/.test(line)) {
      blocks.push(<hr key={nextKey()} className="my-4 border-gray-800" />)
      i++
      continue
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line)
    if (heading) {
      blocks.push(renderHeading(heading[1].length, heading[2], nextKey()))
      i++
      continue
    }

    if (line.startsWith('```')) {
      const content: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) {
        content.push(lines[i])
        i++
      }
      i++
      blocks.push(
        <pre
          key={nextKey()}
          className="bg-gray-900 border border-gray-800 rounded p-3 my-3 overflow-x-auto text-xs text-gray-300 font-mono leading-relaxed"
        >
          <code>{content.join('\n')}</code>
        </pre>
      )
      continue
    }

    if (
      line.startsWith('|') &&
      i + 1 < lines.length &&
      /^\|[\s:|-]+\|?\s*$/.test(lines[i + 1])
    ) {
      const header = parseTableRow(line)
      const rows: string[][] = []
      i += 2
      while (i < lines.length && lines[i].startsWith('|')) {
        rows.push(parseTableRow(lines[i]))
        i++
      }
      blocks.push(renderTable(header, rows, nextKey()))
      continue
    }

    if (/^[-*]\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*]\s+/, ''))
        i++
      }
      blocks.push(
        <ul
          key={nextKey()}
          className="list-disc list-outside ml-5 my-2 space-y-1 text-sm text-gray-300"
        >
          {items.map((item, idx) => (
            <li key={idx} className="leading-relaxed">
              {renderInline(item)}
            </li>
          ))}
        </ul>
      )
      continue
    }

    if (line.trim() === '') {
      i++
      continue
    }

    const paraLines: string[] = []
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !isBlockStart(lines[i], lines[i + 1])
    ) {
      paraLines.push(lines[i])
      i++
    }
    blocks.push(
      <p key={nextKey()} className="my-2 text-sm text-gray-300 leading-relaxed">
        {renderInline(paraLines.join(' '))}
      </p>
    )
  }

  return blocks
}

function isBlockStart(line: string, next?: string): boolean {
  if (/^#{1,6}\s/.test(line)) return true
  if (line.startsWith('```')) return true
  if (/^[-*]\s+/.test(line)) return true
  if (/^---+\s*$/.test(line)) return true
  if (line.startsWith('|') && next !== undefined && /^\|[\s:|-]+\|?\s*$/.test(next)) return true
  return false
}

function parseTableRow(line: string): string[] {
  return line.replace(/^\||\|$/g, '').split('|').map((s) => s.trim())
}

function renderTable(header: string[], rows: string[][], key: number): ReactNode {
  return (
    <div key={key} className="my-3 overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-gray-700">
            {header.map((cell, j) => (
              <th
                key={j}
                className="text-left font-semibold px-2 py-1.5 text-gray-200 align-bottom"
              >
                {renderInline(cell)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r} className="border-b border-gray-900">
              {row.map((cell, c) => (
                <td key={c} className="px-2 py-1 text-gray-400 align-top">
                  {renderInline(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function renderHeading(level: number, text: string, key: number): ReactNode {
  const inline = renderInline(text)
  switch (level) {
    case 1:
      return (
        <h1 key={key} className="text-2xl font-bold text-gray-100 mt-5 mb-3">
          {inline}
        </h1>
      )
    case 2:
      return (
        <h2 key={key} className="text-xl font-bold text-gray-100 mt-5 mb-2">
          {inline}
        </h2>
      )
    case 3:
      return (
        <h3 key={key} className="text-lg font-bold text-gray-200 mt-5 mb-2">
          {inline}
        </h3>
      )
    case 4:
      return (
        <h4 key={key} className="text-base font-semibold text-gray-200 mt-3 mb-1">
          {inline}
        </h4>
      )
    default:
      return (
        <h5 key={key} className="text-sm font-semibold text-gray-300 mt-2 mb-1">
          {inline}
        </h5>
      )
  }
}

function renderInline(text: string): ReactNode {
  const out: ReactNode[] = []
  let buffer = ''
  let i = 0
  const flush = () => {
    if (buffer) {
      out.push(buffer)
      buffer = ''
    }
  }

  while (i < text.length) {
    const c = text[i]

    if (c === '`') {
      const end = text.indexOf('`', i + 1)
      if (end !== -1) {
        flush()
        out.push(
          <code
            key={out.length}
            className="bg-gray-800 text-emerald-300 px-1 py-0.5 rounded font-mono text-[0.85em]"
          >
            {text.slice(i + 1, end)}
          </code>
        )
        i = end + 1
        continue
      }
    }

    if (c === '*' && text[i + 1] === '*') {
      const end = text.indexOf('**', i + 2)
      if (end !== -1) {
        flush()
        out.push(
          <strong key={out.length} className="font-semibold text-gray-100">
            {renderInline(text.slice(i + 2, end))}
          </strong>
        )
        i = end + 2
        continue
      }
    }

    if (c === '*' && text[i + 1] !== '*') {
      const end = text.indexOf('*', i + 1)
      if (end !== -1 && text[end + 1] !== '*') {
        flush()
        out.push(
          <em key={out.length} className="italic">
            {renderInline(text.slice(i + 1, end))}
          </em>
        )
        i = end + 1
        continue
      }
    }

    if (c === '[') {
      const closeBracket = text.indexOf(']', i + 1)
      if (closeBracket !== -1 && text[closeBracket + 1] === '(') {
        const closeParen = text.indexOf(')', closeBracket + 2)
        if (closeParen !== -1) {
          flush()
          const label = text.slice(i + 1, closeBracket)
          const url = text.slice(closeBracket + 2, closeParen)
          out.push(
            <a
              key={out.length}
              href={url}
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noreferrer"
            >
              {label}
            </a>
          )
          i = closeParen + 1
          continue
        }
      }
    }

    buffer += c
    i++
  }
  flush()
  return <>{out}</>
}
