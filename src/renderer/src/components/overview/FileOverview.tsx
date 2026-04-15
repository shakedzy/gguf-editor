import { useFileStore } from '../../store/file-store'
import { formatBytes, formatNumber, formatHex } from '../../lib/format'

export default function FileOverview() {
  const { fileInfo } = useFileStore()
  if (!fileInfo) return null

  const modelName =
    fileInfo.metadata.find((m) => m.key === 'general.name')?.value ?? 'Unknown'
  const arch =
    fileInfo.metadata.find((m) => m.key === 'general.architecture')?.value ?? 'Unknown'
  const fileType =
    fileInfo.metadata.find((m) => m.key === 'general.file_type')?.value

  const cards = [
    { label: 'Model Name', value: String(modelName) },
    { label: 'Architecture', value: String(arch) },
    { label: 'File Type', value: fileType != null ? String(fileType) : 'N/A' },
    { label: 'Magic', value: formatHex(fileInfo.header.magic) },
    { label: 'Version', value: String(fileInfo.header.version) },
    { label: 'File Size', value: formatBytes(fileInfo.fileSize) },
    { label: 'Tensors', value: formatNumber(fileInfo.header.tensorCount) },
    { label: 'Metadata Keys', value: formatNumber(fileInfo.header.metadataKvCount) },
    { label: 'Alignment', value: `${fileInfo.alignment} bytes` },
    {
      label: 'Data Section Offset',
      value: formatHex(fileInfo.dataStartOffset)
    },
    {
      label: 'Data Section Size',
      value: formatBytes(fileInfo.fileSize - fileInfo.dataStartOffset)
    }
  ]

  // Find architecture-specific parameters
  const archParams = fileInfo.metadata
    .filter(
      (m) =>
        m.key.startsWith(`${arch}.`) ||
        m.key.startsWith('general.') ||
        m.key.startsWith('tokenizer.')
    )
    .filter(
      (m) =>
        !m.key.includes('tokenizer.ggml.tokens') &&
        !m.key.includes('tokenizer.ggml.scores') &&
        !m.key.includes('tokenizer.ggml.token_type') &&
        !m.key.includes('tokenizer.ggml.merges')
    )
    .filter((m) => typeof m.value !== 'object')
    .slice(0, 20)

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-xl font-bold text-gray-100">File Overview</h2>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
        {cards.map(({ label, value }) => (
          <div
            key={label}
            className="bg-gray-900 rounded-lg p-4 border border-gray-800"
          >
            <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
              {label}
            </div>
            <div className="text-sm font-medium text-gray-200 font-mono truncate">
              {value}
            </div>
          </div>
        ))}
      </div>

      {archParams.length > 0 && (
        <>
          <h3 className="text-lg font-semibold text-gray-200 mt-6">
            Model Parameters
          </h3>
          <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left px-4 py-2 text-xs text-gray-500 uppercase">
                    Key
                  </th>
                  <th className="text-left px-4 py-2 text-xs text-gray-500 uppercase">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody>
                {archParams.map((kv) => (
                  <tr
                    key={kv.key}
                    className="border-b border-gray-800/50 hover:bg-gray-800/30"
                  >
                    <td className="px-4 py-2 font-mono text-gray-400">
                      {kv.key}
                    </td>
                    <td className="px-4 py-2 font-mono text-gray-200">
                      {String(kv.value)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="text-xs text-gray-600 mt-4">
        File: {fileInfo.filePath}
      </div>
    </div>
  )
}
