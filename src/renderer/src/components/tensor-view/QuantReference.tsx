import { GGML_TYPE_NAME } from '../../lib/tensor-types'
import { getReferenceSection } from '../../lib/quant-reference'
import { renderMarkdown } from '../../lib/markdown'

interface Props {
  type: number
}

export default function QuantReference({ type }: Props) {
  const typeName = GGML_TYPE_NAME[type]
  const content = typeName ? getReferenceSection(typeName) : null

  if (!content) {
    return (
      <div className="p-6 text-sm text-gray-500">
        No reference documentation available for type{' '}
        <code className="font-mono">{typeName ?? `#${type}`}</code>.
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl px-6 py-4">{renderMarkdown(content)}</div>
    </div>
  )
}
