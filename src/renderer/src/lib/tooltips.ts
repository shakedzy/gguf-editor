// Tooltips for overview cards
export const OVERVIEW_TOOLTIPS: Record<string, string> = {
  'Model Name':
    'The human-readable name of this model, stored in general.name metadata.',
  Architecture:
    'The model architecture type (e.g. llama, gemma, phi, clip). Determines which config keys to expect and how layers are structured.',
  'File Type':
    'The quantization type ID used for most tensors in this file. Maps to quant methods like Q4_0, Q4_K_M, etc.',
  Magic:
    'The 4-byte magic number at the start of every GGUF file. Must be 0x46554747 ("GGUF" in ASCII). Used to verify the file is valid GGUF.',
  Version:
    'The GGUF format version. Version 3 is current and supports both little-endian and big-endian layouts.',
  'File Size': 'Total size of the GGUF file on disk.',
  Tensors:
    'Number of weight tensors stored in this file. Each tensor contains a multi-dimensional array of (possibly quantized) values.',
  'Metadata Keys':
    'Number of key-value pairs in the metadata section. Metadata stores model config, tokenizer data, and other non-weight information.',
  Alignment:
    'Byte alignment boundary for tensor data. After the header/metadata section, padding bytes are inserted so tensor data starts at a multiple of this value. Default is 32 bytes. Proper alignment enables efficient memory-mapped access and SIMD operations.',
  'Data Section Offset':
    'The byte offset where tensor data begins in the file. Everything before this is header + metadata + tensor descriptors + alignment padding.',
  'Data Section Size':
    'Total size of all tensor weight data. This is file_size minus the header/metadata overhead.'
}

// Tooltips for well-known GGUF metadata keys
export const METADATA_KEY_TOOLTIPS: Record<string, string> = {
  // General
  'general.architecture':
    'Model architecture identifier (e.g. "llama", "gemma", "phi"). Determines which architecture-specific keys are expected.',
  'general.name': 'Human-readable model name.',
  'general.author': 'Model author or organization.',
  'general.url': 'URL to the model page or repository.',
  'general.description': 'Free-form description of the model.',
  'general.license': 'License identifier (e.g. "apache-2.0", "mit").',
  'general.license.link': 'URL to the full license text.',
  'general.file_type':
    'Quantization type ID for the majority of tensors. 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 7=Q8_0, etc.',
  'general.quantization_version':
    'Version of the quantization scheme used. Helps ensure compatibility.',
  'general.alignment':
    'Byte alignment for tensor data. Tensor offsets must be multiples of this value.',
  'general.type':
    'Type of GGUF file (e.g. "model", "adapter", "mmproj" for multimodal projectors).',
  'general.size_label':
    'Human-readable model size label (e.g. "7B", "13B", "70B").',
  'general.tags': 'Tags for categorizing the model (e.g. "text-generation").',
  'general.sampling.top_k':
    'Default top-k sampling parameter. Limits token selection to top k most likely tokens.',
  'general.sampling.top_p':
    'Default top-p (nucleus) sampling parameter. Limits selection to tokens within cumulative probability p.',
  'general.sampling.temp':
    'Default sampling temperature. Higher = more random, lower = more deterministic.',

  // LLaMA / common architecture keys
  'llama.embedding_length':
    'Dimension of the embedding vectors (hidden size). Each token is represented as a vector of this length.',
  'llama.block_count':
    'Number of transformer blocks (layers) in the model. More layers = more capacity but slower inference.',
  'llama.feed_forward_length':
    'Size of the feed-forward network intermediate layer. Typically 4x the embedding length.',
  'llama.attention.head_count':
    'Number of attention heads. Attention is split into this many parallel heads.',
  'llama.attention.head_count_kv':
    'Number of key/value heads for grouped-query attention (GQA). Fewer KV heads = less memory for KV cache.',
  'llama.attention.layer_norm_rms_epsilon':
    'Epsilon for RMS layer normalization. Prevents division by zero.',
  'llama.rope.freq_base':
    'Base frequency for Rotary Position Embeddings (RoPE). Higher = supports longer context windows.',
  'llama.rope.dimension_count':
    'Number of dimensions used for RoPE. Usually half of head dimension.',
  'llama.context_length':
    'Maximum supported context length (in tokens) for this model.',
  'llama.vocab_size':
    'Size of the tokenizer vocabulary. Number of unique tokens the model can understand.',
  'llama.attention.key_length':
    'Dimension of attention key vectors per head.',
  'llama.attention.value_length':
    'Dimension of attention value vectors per head.',
  'llama.expert_count':
    'Number of experts in a Mixture-of-Experts (MoE) model.',
  'llama.expert_used_count':
    'Number of experts activated per token in MoE routing.',

  // CLIP / Vision
  'clip.has_vision_encoder':
    'Whether this file contains a vision encoder (for multimodal models).',
  'clip.vision.projection_dim':
    'Dimension of the vision-to-text projection output.',
  'clip.vision.image_size':
    'Input image resolution in pixels (square). Images are resized to this before processing.',
  'clip.vision.patch_size':
    'Size of each image patch in pixels. The image is divided into (image_size/patch_size)^2 patches.',
  'clip.vision.embedding_length':
    'Dimension of vision embedding vectors.',
  'clip.vision.feed_forward_length':
    'Feed-forward intermediate size in vision transformer blocks.',
  'clip.vision.block_count':
    'Number of transformer blocks in the vision encoder.',
  'clip.vision.attention.head_count':
    'Number of attention heads in the vision transformer.',
  'clip.vision.image_mean':
    'Per-channel mean values for image normalization (RGB).',
  'clip.vision.image_std':
    'Per-channel standard deviation values for image normalization (RGB).',
  'clip.vision.projector_type':
    'Type of vision projector (e.g. "mlp", "ldp", "gemma4v").',
  'clip.vision.attention.layer_norm_epsilon':
    'Epsilon for layer normalization in vision attention.',

  // Tokenizer
  'tokenizer.ggml.model':
    'Tokenizer model type (e.g. "llama" for SentencePiece, "gpt2" for BPE).',
  'tokenizer.ggml.tokens':
    'Array of all token strings in the vocabulary. Index = token ID.',
  'tokenizer.ggml.scores':
    'Log-probability scores for each token. Used in SentencePiece tokenization.',
  'tokenizer.ggml.token_type':
    'Type of each token: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte.',
  'tokenizer.ggml.merges':
    'BPE merge rules as "token1 token2" strings. Applied in order during tokenization.',
  'tokenizer.ggml.bos_token_id':
    'Token ID for Beginning-of-Sequence. Prepended to every prompt.',
  'tokenizer.ggml.eos_token_id':
    'Token ID for End-of-Sequence. Signals the model to stop generating.',
  'tokenizer.ggml.padding_token_id':
    'Token ID used for padding sequences to equal length in batches.',
  'tokenizer.ggml.unknown_token_id':
    'Token ID for unknown/out-of-vocabulary tokens.',
  'tokenizer.ggml.add_bos_token':
    'Whether to automatically add BOS token at the start of input.',
  'tokenizer.ggml.add_eos_token':
    'Whether to automatically add EOS token at the end of input.',
  'tokenizer.chat_template':
    'Jinja2 template string for formatting chat messages into the model prompt format.'
}

// Try to match a key to a tooltip, including architecture-agnostic matching
export function getMetadataTooltip(key: string): string | null {
  // Direct match
  if (METADATA_KEY_TOOLTIPS[key]) return METADATA_KEY_TOOLTIPS[key]

  // Try replacing architecture prefix with "llama" for generic matching
  // e.g. "gemma.embedding_length" -> "llama.embedding_length"
  const dotIndex = key.indexOf('.')
  if (dotIndex > 0) {
    const suffix = key.slice(dotIndex + 1)
    const llamaKey = `llama.${suffix}`
    if (METADATA_KEY_TOOLTIPS[llamaKey]) {
      return METADATA_KEY_TOOLTIPS[llamaKey]
    }
  }

  return null
}
