import { useMemo, useRef, useState, useCallback, useEffect } from 'react'
import { useFileStore, GgufTensorInfo, GgufFileInfo } from '../../store/file-store'
import { GGML_TYPE_NAME, getTypeBadgeColor } from '../../lib/tensor-types'
import { formatBytes, formatShape } from '../../lib/format'

// ── Parsing tensor names to infer architecture ──────────────────────

interface ModelBlock {
  id: string
  label: string
  type: 'input' | 'block' | 'output' | 'norm' | 'embed' | 'proj' | 'moe' | 'router' | 'expert' | 'other'
  tensors: GgufTensorInfo[]
  children?: ModelBlock[]
  parallel?: boolean // render children side-by-side instead of stacked
}

function isMoeModel(fileInfo: GgufFileInfo): boolean {
  return fileInfo.metadata.some(
    (m) => m.key.endsWith('.expert_count') && typeof m.value === 'number' && m.value > 1
  )
}

function getMoeInfo(fileInfo: GgufFileInfo): { expertCount: number; usedCount: number } {
  const ec = fileInfo.metadata.find((m) => m.key.endsWith('.expert_count'))
  const uc = fileInfo.metadata.find((m) => m.key.endsWith('.expert_used_count'))
  return {
    expertCount: typeof ec?.value === 'number' ? ec.value : 0,
    usedCount: typeof uc?.value === 'number' ? uc.value : 0
  }
}

function inferArchitecture(fileInfo: GgufFileInfo, matchers?: [RegExp, string][]): ModelBlock[] {
  const tensors = fileInfo.tensors
  const blocks: ModelBlock[] = []
  const byPrefix = new Map<string, GgufTensorInfo[]>()

  for (const t of tensors) {
    const parts = t.name.split('.')
    let prefix: string
    // v.blk.N.xxx → prefix = "v.blk.N"
    if (parts[0] === 'v' && parts[1] === 'blk') {
      prefix = parts.slice(0, 3).join('.')
    } else if (parts[0] === 'blk' || parts[0] === 'v' || parts[0] === 'mm') {
      prefix = parts.slice(0, 2).join('.')
    } else {
      prefix = parts[0]
    }
    if (!byPrefix.has(prefix)) byPrefix.set(prefix, [])
    byPrefix.get(prefix)!.push(t)
  }

  const sortedPrefixes = [...byPrefix.keys()].sort((a, b) => {
    const order = (k: string) => {
      if (k.startsWith('token_embd') || k.startsWith('embed') || k === 'mm.0') return 0
      if (k.startsWith('v.blk.')) {
        const num = parseInt(k.split('.')[2])
        return 100 + (isNaN(num) ? 0 : num)
      }
      if (k.startsWith('blk.') || k.startsWith('v.') || k.startsWith('mm.')) {
        const num = parseInt(k.split('.')[1])
        return 100 + (isNaN(num) ? 0 : num)
      }
      if (k.startsWith('output_norm') || k === 'norm') return 90000
      if (k.startsWith('output') || k === 'lm_head') return 90001
      return 50000
    }
    return order(a) - order(b)
  })

  const moe = isMoeModel(fileInfo)
  const moeInfo = moe ? getMoeInfo(fileInfo) : null

  for (const prefix of sortedPrefixes) {
    const groupTensors = byPrefix.get(prefix)!
    blocks.push(classifyBlock(prefix, groupTensors, moe, moeInfo, matchers))
  }
  return blocks
}

function classifyBlock(
  prefix: string,
  tensors: GgufTensorInfo[],
  isMoe: boolean,
  moeInfo: { expertCount: number; usedCount: number } | null,
  matchers?: [RegExp, string][]
): ModelBlock {
  if (prefix.startsWith('token_embd') || prefix.startsWith('embed'))
    return { id: prefix, label: 'Token Embedding', type: 'embed', tensors, children: inferSubComponents(tensors, isMoe, moeInfo, matchers) }
  if (prefix.startsWith('blk.')) {
    const num = prefix.split('.')[1]
    const hasMoeFFN = isMoe && tensors.some((t) => t.name.includes('_exps') || t.name.includes('gate_inp'))
    const label = hasMoeFFN ? `MoE Block ${num}` : `Transformer Block ${num}`
    return { id: prefix, label, type: hasMoeFFN ? 'moe' : 'block', tensors, children: inferSubComponents(tensors, isMoe, moeInfo, matchers) }
  }
  if (prefix.startsWith('v.blk.')) {
    const num = prefix.split('.')[2]
    return { id: prefix, label: `Vision Block ${num}`, type: 'block', tensors, children: inferSubComponents(tensors, false, null, matchers) }
  }
  if (prefix.startsWith('v.'))
    return { id: prefix, label: prefix, type: 'embed', tensors }
  if (prefix.startsWith('mm.') || prefix === 'mm') {
    const parts = prefix.split('.')
    return { id: prefix, label: parts.length > 1 ? `MM Projector ${parts[1]}` : 'MM Projector', type: 'proj', tensors, children: inferSubComponents(tensors, false, null, matchers) }
  }
  if (prefix.includes('norm'))
    return { id: prefix, label: 'Final Norm', type: 'norm', tensors }
  if (prefix === 'output' || prefix === 'lm_head')
    return { id: prefix, label: 'Output / LM Head', type: 'output', tensors }
  return { id: prefix, label: prefix, type: 'other', tensors, children: inferSubComponents(tensors, false, null, matchers) }
}

// ── Flow-ordered classification ─────────────────────────────────────

// Stage priority determines the order sub-blocks appear in the diagram.
const STAGE_ORDER: Record<string, number> = {
  'attn-pre-norm': 0,
  'attn-qkv': 1,
  'attn-output': 2,
  'attn-post-norm': 3,
  'ffn-pre-norm': 4,
  'router': 5,
  'experts': 6,
  'shared-ffn': 7,
  'ffn': 7,
  'ffn-post-norm': 8,
  'layer-scale': 9,
  'other': 10
}

const STAGE_LABELS: Record<string, string> = {
  'attn-pre-norm': 'Pre-attention Norm',
  'attn-qkv': 'QKV Projections',
  'attn-output': 'Attention Output',
  'attn-post-norm': 'Post-attention Norm',
  'ffn-pre-norm': 'Pre-FFN Norm',
  'router': 'Router / Gate',
  'experts': 'Experts',
  'shared-ffn': 'Shared FFN',
  'ffn': 'Feed-Forward',
  'ffn-post-norm': 'Post-FFN Norm',
  'layer-scale': 'Layer Scale',
  'other': 'Other'
}

const STAGE_TYPES: Record<string, ModelBlock['type']> = {
  'attn-pre-norm': 'norm',
  'attn-post-norm': 'norm',
  'ffn-pre-norm': 'norm',
  'ffn-post-norm': 'norm',
  'router': 'router',
  'experts': 'expert',
  'layer-scale': 'other'
}

function classifyTensorSubComponent(name: string, isMoe: boolean): string {
  const parts = name.split('.')

  for (const p of parts) {
    // Norms — classify by position in the flow
    if (p === 'attn_norm' || p === 'ln1') return 'attn-pre-norm'
    if (p === 'post_attention_norm' || p === 'attn_post_norm') return 'attn-post-norm'
    if (p === 'ffn_norm' || p === 'ln2' || p.startsWith('pre_ffw_norm')) return 'ffn-pre-norm'
    if (p.startsWith('post_ffw_norm') || p === 'ffn_post_norm') return 'ffn-post-norm'

    // Attention
    if (p === 'attn_q' || p === 'attn_k' || p === 'attn_v' ||
        p === 'attn_q_norm' || p === 'attn_k_norm') return 'attn-qkv'
    if (p === 'attn_output' || p === 'attn_out') return 'attn-output'

    // MoE routing
    if (p === 'ffn_gate_inp' || p === 'gate_inp') return 'router'
    // MoE expert weights
    if (p.includes('_exps') || p.includes('expert')) return 'experts'
    // In MoE: ffn_gate/up/down without _exps are the shared dense FFN
    if (isMoe && (p === 'ffn_gate' || p === 'ffn_up' || p === 'ffn_down')) return 'shared-ffn'
    // Dense FFN
    if (p.startsWith('ffn') && !p.includes('norm')) return 'ffn'

    // Layer output scale
    if (p === 'layer_output_scale') return 'layer-scale'
  }
  return 'other'
}

// ── Registry-based classification ───────────────────────────────────

// Map LLM_TENSOR_* roles to flow stages
const ROLE_TO_STAGE: Record<string, string> = {
  LLM_TENSOR_ATTN_NORM: 'attn-pre-norm',
  LLM_TENSOR_ATTN_NORM_2: 'attn-pre-norm',
  LLM_TENSOR_ATTN_Q: 'attn-qkv',
  LLM_TENSOR_ATTN_K: 'attn-qkv',
  LLM_TENSOR_ATTN_V: 'attn-qkv',
  LLM_TENSOR_ATTN_QKV: 'attn-qkv',
  LLM_TENSOR_ATTN_Q_NORM: 'attn-qkv',
  LLM_TENSOR_ATTN_K_NORM: 'attn-qkv',
  LLM_TENSOR_ATTN_Q_A: 'attn-qkv',
  LLM_TENSOR_ATTN_Q_B: 'attn-qkv',
  LLM_TENSOR_ATTN_KV_A_MQA: 'attn-qkv',
  LLM_TENSOR_ATTN_KV_B: 'attn-qkv',
  LLM_TENSOR_ATTN_K_B: 'attn-qkv',
  LLM_TENSOR_ATTN_V_B: 'attn-qkv',
  LLM_TENSOR_ATTN_Q_A_NORM: 'attn-qkv',
  LLM_TENSOR_ATTN_KV_A_NORM: 'attn-qkv',
  LLM_TENSOR_ATTN_OUT: 'attn-output',
  LLM_TENSOR_ATTN_POST_NORM: 'attn-post-norm',
  LLM_TENSOR_ATTN_OUT_NORM: 'attn-post-norm',
  LLM_TENSOR_POST_ATTN_NORM: 'attn-post-norm',
  LLM_TENSOR_FFN_NORM: 'ffn-pre-norm',
  LLM_TENSOR_FFN_PRE_NORM_2: 'ffn-pre-norm',
  LLM_TENSOR_FFN_GATE_INP: 'router',
  LLM_TENSOR_FFN_GATE_INP_SHEXP: 'router',
  LLM_TENSOR_FFN_DOWN_EXP: 'experts',
  LLM_TENSOR_FFN_GATE_EXP: 'experts',
  LLM_TENSOR_FFN_UP_EXP: 'experts',
  LLM_TENSOR_FFN_DOWN_EXPS: 'experts',
  LLM_TENSOR_FFN_GATE_EXPS: 'experts',
  LLM_TENSOR_FFN_UP_EXPS: 'experts',
  LLM_TENSOR_FFN_GATE_UP_EXPS: 'experts',
  LLM_TENSOR_FFN_NORM_EXPS: 'experts',
  LLM_TENSOR_FFN_GATE: 'ffn',
  LLM_TENSOR_FFN_DOWN: 'ffn',
  LLM_TENSOR_FFN_UP: 'ffn',
  LLM_TENSOR_FFN_ACT: 'ffn',
  LLM_TENSOR_FFN_DOWN_SHEXP: 'ffn',
  LLM_TENSOR_FFN_GATE_SHEXP: 'ffn',
  LLM_TENSOR_FFN_UP_SHEXP: 'ffn',
  LLM_TENSOR_FFN_POST_NORM: 'ffn-post-norm',
  LLM_TENSOR_FFN_POST_NORM_1: 'ffn-post-norm',
  LLM_TENSOR_FFN_POST_NORM_2: 'ffn-post-norm',
  LLM_TENSOR_POST_MLP_NORM: 'ffn-post-norm',
  LLM_TENSOR_LAYER_OUT_NORM: 'ffn-post-norm',
  LLM_TENSOR_LAYER_OUT_SCALE: 'layer-scale',
  LLM_TENSOR_ATTN_SUB_NORM: 'attn-post-norm',
  LLM_TENSOR_FFN_SUB_NORM: 'ffn-post-norm',
}

// Build regex matchers from registry patterns (computed once)
function buildRegistryMatchers(tensorRoles: Record<string, string>): [RegExp, string][] {
  return Object.entries(tensorRoles).map(([pattern, role]) => {
    const regexStr = '^' + pattern.replace(/%d/g, '\\d+') + '(\\.(weight|bias|scale))?$'
    return [new RegExp(regexStr), role]
  })
}

function classifyByRegistry(
  name: string,
  matchers: [RegExp, string][],
  isMoe: boolean
): string | null {
  for (const [regex, role] of matchers) {
    if (regex.test(name)) {
      // Map role to stage
      let stage = ROLE_TO_STAGE[role]
      if (!stage) return null

      // For MoE: reclassify plain FFN as shared-ffn when experts coexist
      if (isMoe && stage === 'ffn' &&
          (role === 'LLM_TENSOR_FFN_GATE' || role === 'LLM_TENSOR_FFN_UP' || role === 'LLM_TENSOR_FFN_DOWN')) {
        stage = 'shared-ffn'
      }
      return stage
    }
  }
  return null
}

// Sort tensors within a sub-block by computation order
function tensorSortKey(name: string): number {
  const parts = name.split('.')
  for (const p of parts) {
    // QKV order: Q → Q_norm → K → K_norm → V
    if (p === 'attn_q') return 0
    if (p === 'attn_q_norm') return 1
    if (p === 'attn_k') return 2
    if (p === 'attn_k_norm') return 3
    if (p === 'attn_v') return 4
    if (p === 'attn_output' || p === 'attn_out') return 0
    // FFN order: gate → up → down
    if (p === 'ffn_gate' || p === 'ffn_gate_inp') return 0
    if (p === 'ffn_up') return 1
    if (p === 'ffn_down') return 2
    // Expert order: gate_up before down
    if (p === 'ffn_gate_up_exps') return 0
    if (p === 'ffn_down_exps') return 1
  }
  // Scale/bias after weight
  if (name.endsWith('.scale')) return 50
  if (name.endsWith('.bias')) return 51
  return 10
}

function inferSubComponents(
  tensors: GgufTensorInfo[],
  isMoe: boolean,
  moeInfo: { expertCount: number; usedCount: number } | null,
  matchers?: [RegExp, string][]
): ModelBlock[] | undefined {
  if (tensors.length <= 2) return undefined

  // Classify each tensor into a flow stage
  const groups = new Map<string, GgufTensorInfo[]>()
  for (const t of tensors) {
    const key = (matchers && classifyByRegistry(t.name, matchers, isMoe))
      || classifyTensorSubComponent(t.name, isMoe)
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(t)
  }
  if (groups.size <= 1) return undefined

  // Sort tensors within each group by computation order
  for (const g of groups.values()) {
    g.sort((a, b) => tensorSortKey(a.name) - tensorSortKey(b.name))
  }

  // Build sub-blocks in flow order
  const result: ModelBlock[] = []
  const orderedStages = [...groups.keys()].sort(
    (a, b) => (STAGE_ORDER[a] ?? 99) - (STAGE_ORDER[b] ?? 99)
  )

  // Check if we should create a parallel MoE group
  const hasRouter = groups.has('router')
  const hasExperts = groups.has('experts')
  const hasSharedFfn = groups.has('shared-ffn')
  const isMoeBlock = hasRouter && hasExperts

  for (const stage of orderedStages) {
    // For MoE: group router + experts + shared-ffn into a parallel block
    if (isMoeBlock && (stage === 'router' || stage === 'experts' || stage === 'shared-ffn')) {
      // Only build the parallel group once (when we hit the first of the three)
      if (stage !== 'router') continue

      const expertPath: ModelBlock[] = []
      expertPath.push({ id: 'router', label: 'Router / Gate', type: 'router', tensors: groups.get('router')! })
      if (hasExperts) {
        expertPath.push({
          id: 'experts',
          label: `Experts (${moeInfo?.expertCount ?? '?'} total, top-${moeInfo?.usedCount ?? '?'} routed)`,
          type: 'expert',
          tensors: groups.get('experts')!
        })
      }

      const children: ModelBlock[] = [
        { id: 'expert-path', label: 'Expert Path', type: 'other', tensors: [...(groups.get('router') ?? []), ...(groups.get('experts') ?? [])], children: expertPath }
      ]
      if (hasSharedFfn) {
        children.push({ id: 'shared-ffn', label: 'Shared FFN', type: 'other', tensors: groups.get('shared-ffn')! })
      }

      result.push({
        id: 'moe-ffn',
        label: 'MoE FFN',
        type: 'moe',
        tensors: [...(groups.get('router') ?? []), ...(groups.get('experts') ?? []), ...(groups.get('shared-ffn') ?? [])],
        children,
        parallel: true
      })
      continue
    }

    const g = groups.get(stage)!
    const stageType = STAGE_TYPES[stage] ?? 'other'

    if (stage === 'experts' && !isMoeBlock && moeInfo) {
      result.push({
        id: stage,
        label: `Experts (${moeInfo.expertCount} total, top-${moeInfo.usedCount} routed)`,
        type: 'expert',
        tensors: g
      })
    } else {
      result.push({
        id: stage,
        label: STAGE_LABELS[stage] ?? stage,
        type: stageType,
        tensors: g
      })
    }
  }

  return result
}

// ── Color scheme ────────────────────────────────────────────────────

const TYPE_COLORS: Record<string, { bg: string; border: string; text: string; accent: string }> = {
  embed:   { bg: 'bg-emerald-950/60', border: 'border-emerald-700/50', text: 'text-emerald-300', accent: 'bg-emerald-500' },
  block:   { bg: 'bg-blue-950/60',    border: 'border-blue-700/50',    text: 'text-blue-300',    accent: 'bg-blue-500' },
  moe:     { bg: 'bg-indigo-950/60',  border: 'border-indigo-700/50',  text: 'text-indigo-300',  accent: 'bg-indigo-500' },
  norm:    { bg: 'bg-purple-950/60',  border: 'border-purple-700/50',  text: 'text-purple-300',  accent: 'bg-purple-500' },
  output:  { bg: 'bg-rose-950/60',    border: 'border-rose-700/50',    text: 'text-rose-300',    accent: 'bg-rose-500' },
  proj:    { bg: 'bg-amber-950/60',   border: 'border-amber-700/50',   text: 'text-amber-300',   accent: 'bg-amber-500' },
  input:   { bg: 'bg-cyan-950/60',    border: 'border-cyan-700/50',    text: 'text-cyan-300',    accent: 'bg-cyan-500' },
  router:  { bg: 'bg-orange-950/60',  border: 'border-orange-700/50',  text: 'text-orange-300',  accent: 'bg-orange-500' },
  expert:  { bg: 'bg-teal-950/60',    border: 'border-teal-700/50',    text: 'text-teal-300',    accent: 'bg-teal-500' },
  other:   { bg: 'bg-gray-900/60',    border: 'border-gray-700/50',    text: 'text-gray-300',    accent: 'bg-gray-500' }
}

// ── Helper: find which block contains a tensor ──────────────────────

function findBlockForTensor(blocks: ModelBlock[], tensorName: string): string | null {
  for (const b of blocks) {
    if (b.tensors.some((t) => t.name === tensorName)) return b.id
  }
  return null
}

// ── Diagram rendering ───────────────────────────────────────────────

export default function ModelDiagram() {
  const { fileInfo, diagramFocusTensor, clearDiagramFocus } = useFileStore()
  const [expandedBlocks, setExpandedBlocks] = useState<Set<string>>(new Set())
  const [collapsedRange, setCollapsedRange] = useState(true)
  const [registrySource, setRegistrySource] = useState<'live' | 'cached' | 'offline' | 'loading'>('loading')
  const [registryFetchedAt, setRegistryFetchedAt] = useState('')
  const [registryRefreshing, setRegistryRefreshing] = useState(false)
  const [registryMatchers, setRegistryMatchers] = useState<[RegExp, string][] | undefined>(undefined)
  const scrollRef = useRef<HTMLDivElement>(null)
  const blockRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const tensorRefs = useRef<Map<string, HTMLDivElement>>(new Map())

  const applyRegistry = useCallback((result: { tensorRoles: Record<string, string>; source: 'live' | 'cached' | 'offline'; fetchedAt: string }) => {
    setRegistrySource(result.source)
    setRegistryFetchedAt(result.fetchedAt)
    if (result.source !== 'offline' && Object.keys(result.tensorRoles).length > 0) {
      setRegistryMatchers(buildRegistryMatchers(result.tensorRoles))
    }
  }, [])

  // Load arch registry on mount (cache-first)
  useEffect(() => {
    window.api.getArchRegistry().then(applyRegistry).catch(() => setRegistrySource('offline'))
  }, [applyRegistry])

  const handleRefreshRegistry = useCallback(() => {
    setRegistryRefreshing(true)
    window.api.refreshArchRegistry().then((result) => {
      applyRegistry(result)
      setRegistryRefreshing(false)
    }).catch(() => {
      setRegistryRefreshing(false)
    })
  }, [applyRegistry])

  const blocks = useMemo(() => {
    if (!fileInfo) return []
    return inferArchitecture(fileInfo, registryMatchers)
  }, [fileInfo, registryMatchers])

  useEffect(() => {
    if (!diagramFocusTensor || blocks.length === 0) return
    const blockId = findBlockForTensor(blocks, diagramFocusTensor)
    if (!blockId) return

    const transformerBlocks = blocks.filter((b) => b.type === 'block' || b.type === 'moe')
    if (collapsedRange && transformerBlocks.length > 6) {
      const first3 = transformerBlocks.slice(0, 3)
      const last2 = transformerBlocks.slice(-2)
      const visible = [...first3, ...last2]
      const isVisible = visible.some((b) => b.id === blockId) || blocks.some((b) => b.id === blockId && b.type !== 'block' && b.type !== 'moe')
      if (!isVisible) setCollapsedRange(false)
    }

    setExpandedBlocks((prev) => { const next = new Set(prev); next.add(blockId); return next })

    requestAnimationFrame(() => {
      setTimeout(() => {
        const el = tensorRefs.current.get(diagramFocusTensor)
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' })
          el.classList.add('ring-2', 'ring-blue-500', 'ring-offset-1', 'ring-offset-gray-900')
          setTimeout(() => el.classList.remove('ring-2', 'ring-blue-500', 'ring-offset-1', 'ring-offset-gray-900'), 2000)
        } else {
          const blockEl = blockRefs.current.get(blockId)
          if (blockEl) blockEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
        clearDiagramFocus()
      }, 100)
    })
  }, [diagramFocusTensor, blocks])

  const toggleExpand = useCallback((id: string) => {
    setExpandedBlocks((prev) => { const next = new Set(prev); if (next.has(id)) next.delete(id); else next.add(id); return next })
  }, [])

  if (!fileInfo) return null

  const transformerBlocks = blocks.filter((b) => b.type === 'block' || b.type === 'moe')
  const showCollapse = transformerBlocks.length > 6

  let visibleBlocks = blocks
  if (showCollapse && collapsedRange) {
    const first3 = transformerBlocks.slice(0, 3)
    const last2 = transformerBlocks.slice(-2)
    const insertIdx = blocks.indexOf(transformerBlocks[0])
    visibleBlocks = [
      ...blocks.slice(0, insertIdx),
      ...first3,
      { id: '__collapsed__', label: `... ${transformerBlocks.length - 5} more blocks ...`, type: 'block' as const, tensors: [] },
      ...last2,
      ...blocks.slice(blocks.indexOf(transformerBlocks[transformerBlocks.length - 1]) + 1)
    ]
  }

  const totalParams = fileInfo.tensors.reduce((sum, t) => sum + t.dims.reduce((a, b) => a * b, 1), 0)
  const moe = isMoeModel(fileInfo)
  const moeInfo = moe ? getMoeInfo(fileInfo) : null

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-800 shrink-0">
        <h2 className="text-xl font-bold text-gray-100 mb-1">Model Architecture</h2>
        <div className="flex items-center gap-4 text-xs text-gray-500 flex-wrap">
          <span>{blocks.length} components</span>
          <span>{fileInfo.tensors.length} tensors</span>
          <span>~{totalParams > 1e9 ? (totalParams / 1e9).toFixed(1) + 'B' : (totalParams / 1e6).toFixed(1) + 'M'} parameters</span>
          {moeInfo && (
            <span className="text-indigo-400">
              MoE: {moeInfo.expertCount} experts, top-{moeInfo.usedCount} routing
            </span>
          )}
          {showCollapse && (
            <button onClick={() => setCollapsedRange(!collapsedRange)} className="text-blue-400 hover:text-blue-300">
              {collapsedRange ? 'Expand all blocks' : 'Collapse repeated blocks'}
            </button>
          )}
          <span className="flex items-center gap-1.5">
            <span className={registrySource === 'offline' ? 'text-gray-600' : 'text-emerald-600'}>
              {registrySource === 'loading' ? 'loading tensor roles...' :
               registrySource === 'live' ? 'tensor roles: llama.cpp (live) | flow order: inferred' :
               registrySource === 'cached' ? 'tensor roles: llama.cpp (cached) | flow order: inferred' :
               'tensor roles & flow order: inferred (offline)'}
            </span>
            {registrySource !== 'loading' && (
              <button
                onClick={handleRefreshRegistry}
                disabled={registryRefreshing}
                title={registryFetchedAt
                  ? `Last fetched: ${new Date(registryFetchedAt).toLocaleString()}\nClick to refresh from GitHub`
                  : 'Fetch tensor roles from llama.cpp GitHub'}
                className="text-gray-500 hover:text-emerald-400 transition-colors disabled:opacity-40"
              >
                {registryRefreshing ? '\u21bb' : '\u21bb'}
              </button>
            )}
          </span>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-auto p-6">
        <div className="flex flex-col items-center gap-0 max-w-4xl mx-auto">
          <div className="text-xs text-gray-600 mb-1">Input Tokens</div>
          <Arrow />
          {visibleBlocks.map((block, i) => (
            <div key={block.id} className="flex flex-col items-center w-full">
              <div ref={(el) => { if (el) blockRefs.current.set(block.id, el) }} className="w-full">
                <BlockNode
                  block={block}
                  expanded={expandedBlocks.has(block.id)}
                  onToggle={() => toggleExpand(block.id)}
                  isCollapsedPlaceholder={block.id === '__collapsed__'}
                  tensorRefs={tensorRefs}
                  allTensors={fileInfo.tensors}
                />
              </div>
              {i < visibleBlocks.length - 1 && <Arrow />}
            </div>
          ))}
          <Arrow />
          <div className="text-xs text-gray-600 mt-1">Output Logits</div>
        </div>
      </div>
    </div>
  )
}

function Arrow() {
  return (
    <div className="flex flex-col items-center">
      <div className="w-px h-4 bg-gray-700" />
      <div className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-t-[6px] border-t-gray-700" />
    </div>
  )
}

function SmallArrow() {
  return (
    <div className="flex flex-col items-center">
      <div className="w-px h-2 bg-gray-700" />
      <div className="w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[4px] border-t-gray-700" />
    </div>
  )
}

function BlockNode({
  block, expanded, onToggle, isCollapsedPlaceholder, tensorRefs, allTensors
}: {
  block: ModelBlock
  expanded: boolean
  onToggle: () => void
  isCollapsedPlaceholder?: boolean
  tensorRefs: React.MutableRefObject<Map<string, HTMLDivElement>>
  allTensors: GgufTensorInfo[]
}) {
  const colors = TYPE_COLORS[block.type] || TYPE_COLORS.other
  const totalSize = block.tensors.reduce((s, t) => s + t.sizeBytes, 0)
  const totalParams = block.tensors.reduce((s, t) => s + t.dims.reduce((a, b) => a * b, 1), 0)

  if (isCollapsedPlaceholder) {
    return (
      <div className="w-full max-w-3xl mx-auto py-3 text-center">
        <div className="border-2 border-dashed border-gray-700 rounded-lg py-3 px-6 text-sm text-gray-500">{block.label}</div>
      </div>
    )
  }

  return (
    <div className={`w-full max-w-3xl mx-auto rounded-lg border ${colors.border} ${colors.bg} overflow-hidden`}>
      <button onClick={onToggle} className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/5 transition-colors">
        <div className={`w-2 h-2 rounded-full ${colors.accent} shrink-0`} />
        <div className="flex-1 min-w-0">
          <div className={`text-sm font-semibold ${colors.text}`}>{block.label}</div>
          <div className="text-xs text-gray-500 mt-0.5">
            {block.tensors.length} tensor{block.tensors.length !== 1 ? 's' : ''}
            {totalSize > 0 && <span className="ml-2">{formatBytes(totalSize)}</span>}
            {totalParams > 0 && (
              <span className="ml-2">
                {totalParams > 1e6 ? `${(totalParams / 1e6).toFixed(1)}M params` : `${(totalParams / 1e3).toFixed(1)}K params`}
              </span>
            )}
          </div>
        </div>
        <span className="text-gray-600 text-xs shrink-0">{expanded ? 'v' : '>'}</span>
      </button>

      {expanded && (
        <div className="border-t border-gray-800/50 px-4 py-3 space-y-2">
          {block.children && block.children.length > 0 ? (
            <div className="space-y-1">
              {block.children.map((child, ci) => {
                // Check if this child is a parallel container
                if (child.parallel && child.children && child.children.length > 0) {
                  return (
                    <div key={child.id}>
                      <ParallelBlock child={child} tensorRefs={tensorRefs} allTensors={allTensors} />
                      {ci < block.children!.length - 1 && <SmallArrow />}
                    </div>
                  )
                }
                return (
                  <div key={child.id}>
                    <SubBlock child={child} tensorRefs={tensorRefs} allTensors={allTensors} />
                    {ci < block.children!.length - 1 && <SmallArrow />}
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="space-y-0.5">
              {block.tensors.map((t) => (
                <TensorRow key={t.name} tensor={t} tensorRefs={tensorRefs} allTensors={allTensors} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ParallelBlock({
  child, tensorRefs, allTensors
}: {
  child: ModelBlock
  tensorRefs: React.MutableRefObject<Map<string, HTMLDivElement>>
  allTensors: GgufTensorInfo[]
}) {
  const colors = TYPE_COLORS[child.type] || TYPE_COLORS.other
  return (
    <div className={`rounded-md border ${colors.border} ${colors.bg} overflow-hidden`}>
      <div className={`text-xs font-medium ${colors.text} px-3 py-1.5 flex items-center gap-2 border-b border-gray-800/30`}>
        <span className={`w-1.5 h-1.5 rounded-full ${colors.accent}`} />
        {child.label}
        <span className="text-[10px] text-gray-500 ml-1">parallel</span>
      </div>
      <div className="flex gap-px bg-gray-800/30 items-stretch">
        {child.children!.map((branch) => (
          <div key={branch.id} className="flex-1 min-w-0 bg-gray-950/50 p-2">
            {branch.children ? (
              <div className="space-y-1">
                {branch.children.map((sub, si) => (
                  <div key={sub.id}>
                    <SubBlock child={sub} tensorRefs={tensorRefs} allTensors={allTensors} />
                    {si < branch.children!.length - 1 && <SmallArrow />}
                  </div>
                ))}
              </div>
            ) : (
              <SubBlock child={branch} tensorRefs={tensorRefs} allTensors={allTensors} />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function SubBlock({
  child, tensorRefs, allTensors
}: {
  child: ModelBlock
  tensorRefs: React.MutableRefObject<Map<string, HTMLDivElement>>
  allTensors: GgufTensorInfo[]
}) {
  const colors = TYPE_COLORS[child.type] || TYPE_COLORS.other

  // Expert sub-block: show packed tensor shapes with expert dimension highlighted
  if (child.type === 'expert') {
    return (
      <div className={`rounded-md border ${colors.border} ${colors.bg} px-3 py-2`}>
        <div className={`text-xs font-medium ${colors.text} mb-1.5 flex items-center gap-2`}>
          <span className={`w-1.5 h-1.5 rounded-full ${colors.accent}`} />
          {child.label}
        </div>
        {/* Show each expert tensor with shape annotation */}
        <div className="space-y-1.5">
          {child.tensors.map((t) => {
            const parts = t.name.split('.')
            const shortName = parts.length > 2 ? parts.slice(-2).join('.') : t.name
            // Find the expert dimension (typically the last dim matching expertCount)
            const dims = t.dims
            const expertDimIdx = dims.length > 2 ? dims.length - 1 : -1
            const expertCount = expertDimIdx >= 0 ? dims[expertDimIdx] : null
            const perExpertDims = expertDimIdx >= 0 ? dims.slice(0, expertDimIdx) : dims

            const handleClick = () => {
              const idx = allTensors.findIndex((at) => at.name === t.name)
              if (idx >= 0) useFileStore.getState().navigateToTensor(idx)
            }

            return (
              <div
                key={t.name}
                ref={(el) => { if (el) tensorRefs.current.set(t.name, el) }}
                onClick={handleClick}
                className="rounded bg-gray-800/50 border border-gray-700/30 px-2.5 py-1.5 cursor-pointer hover:bg-white/5 transition-colors group"
                title={`Click to view ${t.name} in Tensors tab`}
              >
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-mono text-gray-400 group-hover:text-teal-400 transition-colors truncate">
                    {shortName}
                  </span>
                  <span className={`px-1 py-0.5 rounded text-[10px] font-mono shrink-0 ${getTypeBadgeColor(t.type)}`}>
                    {GGML_TYPE_NAME[t.type] ?? '?'}
                  </span>
                  <span className="text-gray-600 text-[10px] shrink-0">{formatBytes(t.sizeBytes)}</span>
                </div>
                {expertCount && expertCount > 1 ? (
                  <div className="mt-1 text-[10px] font-mono flex items-center gap-1">
                    <span className="text-gray-500">shape:</span>
                    <span className="text-gray-400">{perExpertDims.join(' x ')}</span>
                    <span className="text-gray-600">x</span>
                    <span className="text-teal-400 font-semibold">{expertCount}</span>
                    <span className="text-gray-600 ml-1">
                      ({expertCount} experts, each {perExpertDims.join('x')})
                    </span>
                  </div>
                ) : (
                  <div className="mt-1 text-[10px] font-mono text-gray-500">
                    shape: {formatShape(dims)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  // Router sub-block: show with shape showing hidden_dim -> expert_count
  if (child.type === 'router') {
    return (
      <div className={`rounded-md border ${colors.border} ${colors.bg} px-3 py-2`}>
        <div className={`text-xs font-medium ${colors.text} mb-1.5 flex items-center gap-2`}>
          <span className={`w-1.5 h-1.5 rounded-full ${colors.accent}`} />
          {child.label}
        </div>
        <div className="space-y-1">
          {child.tensors.map((t) => {
            const parts = t.name.split('.')
            const shortName = parts.length > 2 ? parts.slice(-2).join('.') : t.name
            const dims = t.dims
            // For router: shape is typically [hidden_dim, expert_count]
            const isRouterWeight = dims.length === 2 && !t.name.includes('scale')
            const handleClick = () => {
              const idx = allTensors.findIndex((at) => at.name === t.name)
              if (idx >= 0) useFileStore.getState().navigateToTensor(idx)
            }

            return (
              <div
                key={t.name}
                ref={(el) => { if (el) tensorRefs.current.set(t.name, el) }}
                onClick={handleClick}
                className="rounded bg-gray-800/50 border border-gray-700/30 px-2.5 py-1.5 cursor-pointer hover:bg-white/5 transition-colors group"
                title={`Click to view ${t.name} in Tensors tab`}
              >
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-mono text-gray-400 group-hover:text-orange-400 transition-colors truncate">
                    {shortName}
                  </span>
                  <span className={`px-1 py-0.5 rounded text-[10px] font-mono shrink-0 ${getTypeBadgeColor(t.type)}`}>
                    {GGML_TYPE_NAME[t.type] ?? '?'}
                  </span>
                </div>
                {isRouterWeight ? (
                  <div className="mt-1 text-[10px] font-mono flex items-center gap-1">
                    <span className="text-gray-400">{dims[0]}</span>
                    <span className="text-gray-600">{'->'}</span>
                    <span className="text-orange-400 font-semibold">{dims[1]}</span>
                    <span className="text-gray-600 ml-1">
                      (scores {dims[1]} experts, picks top-k)
                    </span>
                  </div>
                ) : (
                  <div className="mt-1 text-[10px] font-mono text-gray-500">
                    shape: {formatShape(dims)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  // Default sub-block
  return (
    <div className="rounded-md bg-gray-800/40 border border-gray-700/30 px-3 py-2">
      <div className="text-xs font-medium text-gray-400 mb-1.5">{child.label}</div>
      <div className="space-y-0.5">
        {child.tensors.map((t) => (
          <TensorRow key={t.name} tensor={t} tensorRefs={tensorRefs} allTensors={allTensors} />
        ))}
      </div>
    </div>
  )
}

function TensorChip({
  tensor, tensorRefs, allTensors
}: {
  tensor: GgufTensorInfo
  tensorRefs: React.MutableRefObject<Map<string, HTMLDivElement>>
  allTensors: GgufTensorInfo[]
}) {
  const parts = tensor.name.split('.')
  const shortName = parts.length > 2 ? parts.slice(-2).join('.') : tensor.name

  const handleClick = () => {
    const idx = allTensors.findIndex((t) => t.name === tensor.name)
    if (idx >= 0) useFileStore.getState().navigateToTensor(idx)
  }

  return (
    <div
      ref={(el) => { if (el) tensorRefs.current.set(tensor.name, el) }}
      onClick={handleClick}
      className="px-2 py-1 rounded bg-gray-800/60 border border-gray-700/40 text-[10px] font-mono text-gray-400 cursor-pointer hover:text-teal-400 hover:border-teal-700/40 transition-colors truncate max-w-[120px]"
      title={`${tensor.name}\n${formatShape(tensor.dims)} | ${GGML_TYPE_NAME[tensor.type] ?? '?'} | ${formatBytes(tensor.sizeBytes)}`}
    >
      {shortName}
    </div>
  )
}

function TensorRow({
  tensor, tensorRefs, allTensors
}: {
  tensor: GgufTensorInfo
  tensorRefs: React.MutableRefObject<Map<string, HTMLDivElement>>
  allTensors: GgufTensorInfo[]
}) {
  const parts = tensor.name.split('.')
  const shortName = parts.length > 2 ? parts.slice(-2).join('.') : tensor.name

  const handleClick = () => {
    const idx = allTensors.findIndex((t) => t.name === tensor.name)
    if (idx >= 0) useFileStore.getState().navigateToTensor(idx)
  }

  return (
    <div
      ref={(el) => { if (el) tensorRefs.current.set(tensor.name, el) }}
      className="flex items-center gap-2 text-xs py-0.5 group cursor-pointer hover:bg-white/5 rounded px-1 -mx-1 transition-colors"
      onClick={handleClick}
      title={`Click to view ${tensor.name} in Tensors tab`}
    >
      <span className="font-mono text-gray-400 truncate flex-1 group-hover:text-blue-400 transition-colors">{shortName}</span>
      <span className="text-gray-600 font-mono shrink-0">{formatShape(tensor.dims)}</span>
      <span className={`px-1 py-0.5 rounded text-[10px] font-mono shrink-0 ${getTypeBadgeColor(tensor.type)}`}>
        {GGML_TYPE_NAME[tensor.type] ?? '?'}
      </span>
      <span className="text-gray-600 text-[10px] w-14 text-right shrink-0">{formatBytes(tensor.sizeBytes)}</span>
    </div>
  )
}
