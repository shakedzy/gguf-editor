# GGUF Quantization Types — Complete Reference

A practical guide to every quantization type supported by GGUF / ggml, with block layouts, bit accounting, dependencies, and recipe composition. Written for people staring at the hex view in `gguf-editor` going "wait, what is this?"

---

## Contents

- [Preliminaries](#preliminaries)
- [Glossary of Symbols](#glossary-of-symbols)
- [Unquantized Types](#unquantized-types)
  - [F32 · F16 · BF16](#f32--f16--bf16)
- [Simple Quants (32 weights/block)](#simple-quants-32-weightsblock)
  - [Q4_0](#q4_0) · [Q4_1](#q4_1) · [Q5_0](#q5_0) · [Q5_1](#q5_1) · [Q8_0](#q8_0) · [Q8_1](#q8_1)
- [K-Quants (256 weights/block)](#k-quants-256-weightsblock)
  - [Q2_K](#q2_k) · [Q3_K](#q3_k) · [Q4_K](#q4_k) · [Q5_K](#q5_k) · [Q6_K](#q6_k) · [Q8_K](#q8_k)
- [IQ-Quants (importance-matrix, 256 weights/block)](#iq-quants-importance-matrix-256-weightsblock)
  - [IQ1_S](#iq1_s) · [IQ1_M](#iq1_m)
  - [IQ2_XXS](#iq2_xxs) · [IQ2_XS](#iq2_xs) · [IQ2_S](#iq2_s)
  - [IQ3_XXS](#iq3_xxs) · [IQ3_S](#iq3_s)
  - [IQ4_NL](#iq4_nl) · [IQ4_XS](#iq4_xs)
- [File-level Presets (llama.cpp recipes)](#file-level-presets-llamacpp-recipes)
- [The Imatrix](#the-imatrix)
- [Comparison Cheatsheet](#comparison-cheatsheet)
- [Choosing a Quant](#choosing-a-quant)

---

## Preliminaries

**Why quantize.** A float16 weight costs 16 bits. A 7B model at f16 is 14 GB; at 4 bits/weight it's 3.5 GB. Quantization trades some numerical accuracy for dramatic reductions in memory and bandwidth, which is the main bottleneck on consumer hardware.

**The universal structure.** Every GGUF quant type follows the same pattern: a tensor is split into fixed-size **blocks** of weights, and each block stores two things:

1. **Metadata** — one or more floating-point scales/minimums that set the block's dynamic range.
2. **Compressed values** — tight integers, nibbles, or lookup-table indices for each weight in the block.

At inference time, you reverse the process per block: read the metadata, read the compressed value, do a small arithmetic formula, out comes a (close to original) float.

**Three families.**

| Family | Block size | Character |
|---|---|---|
| **Simple Q** (`Q4_0`, `Q5_1`, etc.) | 32 weights | One scale per block. Fast, bulky, medium quality. |
| **K-quants** (`Q3_K`, `Q5_K`, etc.) | 256 weights, with 8 or 16 sub-blocks | Super-block scale + per-sub-block scales. Best general-purpose. |
| **IQ-quants** (`IQ3_S`, `IQ4_XS`, etc.) | 256 weights | Use a hardcoded **codebook** (lookup table) baked into ggml. Go below 4 bits/weight gracefully, but require an *importance matrix* for best results. |

**Naming convention.**

- `Q<n>` — `n` bits per weight with linear mapping.
- `_0` / `_1` suffix on simple quants — symmetric (0) or asymmetric (1, has an extra min).
- `_K` suffix — uses the K-quant super-block structure.
- `IQ<n>_...` — importance-matrix quant. Uses a codebook.
- `XXS < XS < S < M < L` suffix — size/quality tier. More letters = smaller file, lower quality.
- `NL` suffix — "non-linear" lookup (only in `IQ4_*`).

---

## Glossary of Symbols

Symbols you'll see throughout the formulas.

| Symbol | Meaning |
|---|---|
| `w_i` | reconstructed float weight at position *i* |
| `d` | **block scale** — a float16 (or float32 for Q8_K) multiplier applied to the whole block |
| `d_min` / `dmin` | block minimum offset — separate float16 paired with `d` in asymmetric K-quants |
| `m` | block minimum (simple Q_1 quants) |
| `sc_j` | **sub-block scale** — smaller-precision scale (4-, 6-, or 8-bit) for each sub-block *j* within a K-quant super-block |
| `m_j` | sub-block minimum |
| `q_i`, `qs_i`, `ql_i` | raw quantized value for weight *i* (nibble / int8 / low bits) |
| `qh_i`, `hmask` | **high bits** — extra bits stored in a separate field and OR'd into `q_i` to form a wider value |
| `idx_i` | codebook index (IQ-quants) — points into `LUT` |
| `LUT` | **lookup table / codebook** — hardcoded inside ggml, not in the GGUF file. Each entry stores a pre-chosen small vector of weights. |
| `NL` | **non-linear** 16-entry float table for `IQ4_NL` / `IQ4_XS` |
| `sign_i` | ±1 sign bit for weight *i* (IQ3_S, IQ3_XXS) |
| `\|` | bitwise OR — used to glue low + high bits into a wider index |

**How a codebook lookup works.** For IQ-types, `LUT[idx]` returns a whole *group* of weights (typically 4 or 8 at a time), not a single value. The codebook is a table of the most statistically common small weight patterns that appear in trained neural networks. See [IQ3_S](#iq3_s) for the canonical walk-through.

---

## Unquantized Types

### F32 · F16 · BF16

| Property | F32 | F16 | BF16 |
|---|---|---|---|
| **ID** | 0 | 1 | 30 |
| **Bits/weight** | 32 | 16 | 16 |
| **Bytes/weight** | 4 | 2 | 2 |
| **Block size** | 1 | 1 | 1 |
| **Range** | ~1.4e-45 … 3.4e38 | ~6e-5 … 65504 | Same range as F32 |
| **Precision** | ~7 decimals | ~3-4 decimals | ~2-3 decimals |
| **Formula** | `value = float32(bytes)` | `value = float16_to_float32(bytes)` | `value = bfloat16_to_float32(bytes)` |

**Summary:** No quantization. Each weight is stored as a standard IEEE-754 float. F32 is full precision; F16 is half; BF16 ("brain float") is half precision with F32's huge exponent range but fewer mantissa bits — great for training, used increasingly for inference too.

**Dependencies:** None.

**When you'll see it:** Embedding tables (`token_embd.weight`) and output heads (`output.weight`) are *often* kept at F16 or F32 even in quantized files because they're more sensitive. Some modern conversion scripts keep BF16 everywhere as the "uncompressed" baseline.

**Comparison:**
- **F16 vs BF16** — same storage (2B), different bit allocation. F16 has 5-bit exponent + 10-bit mantissa (more precision, smaller range). BF16 has 8-bit exponent + 7-bit mantissa (less precision, full F32 range). BF16 rarely overflows during training, which is why modern models train in it.
- **F32 vs F16** — F32 has ~16× the representable precision but 2× the memory. For inference, F16 is almost always fine.

---

## Simple Quants (32 weights/block)

The oldest format family. One scale per 32 weights. Fast on CPU but bulky compared to K-quants of the same bit count.

### Q4_0

**Summary:** The simplest 4-bit quant. Chop the tensor into blocks of 32 weights, record one float16 scale per block, then pack each weight into 4 bits (a "nibble").

**Block layout** — 18 bytes, 32 weights:

```
┌─────────┬─────────────────────────────┐
│ d (2B)  │ nibbles (16B, 2 weights/B)  │
└─────────┴─────────────────────────────┘
```

**Formula:**

```
w_i = d × (q_i − 8)
```

- `q_i` is a 4-bit nibble, value 0–15.
- Subtracting 8 shifts the range to −8…+7 (signed).

**Bits per weight:**

| Source | Bits |
|---|---|
| nibbles | 4.00 |
| scale `d` (16 bits / 32 weights) | 0.50 |
| **Total** | **4.5 bits/weight** |

**Dependencies:** None.

**Comparison:**
- **Q4_0 vs Q4_1** — Q4_0 is symmetric (centred on zero); Q4_1 adds a block minimum `m` so it can represent asymmetric ranges. Q4_1 is slightly larger.
- **Q4_0 vs IQ4_NL** — identical block size. Q4_0 uses linear `(nibble − 8)`. IQ4_NL routes the nibble through a non-linear lookup table tuned for real weight distributions. IQ4_NL is usually noticeably more accurate.
- **Q4_0 vs Q4_K** — Q4_K uses 256-weight blocks with sub-block scales, getting much better quality at a similar bit count. Q4_0 is legacy; prefer Q4_K for anything new.

---

### Q4_1

**Summary:** Like Q4_0, but with an extra "minimum" offset per block so it can handle asymmetric distributions (weights clustered above or below zero).

**Block layout** — 20 bytes, 32 weights:

```
┌────────┬────────┬─────────────────┐
│ d (2B) │ m (2B) │ nibbles (16B)   │
└────────┴────────┴─────────────────┘
```

**Formula:**

```
w_i = d × q_i + m
```

**Bits per weight:** 4 (nibbles) + 0.5 (d) + 0.5 (m) = **5.0 bits/weight**.

**Dependencies:** None.

**Comparison:** See Q4_0 above. Q4_1 adds ~10% size for the `m` field; the quality improvement vs Q4_0 is small and Q4_K beats both easily.

---

### Q5_0

**Summary:** Symmetric 5-bit quant. Can't pack 5 bits per byte neatly, so it stores 4 bits in the main nibble field and the 5th bit in a separate "high bits" byte array.

**Block layout** — 22 bytes, 32 weights:

```
┌────────┬─────────────────┬──────────────────┐
│ d (2B) │ high bits (4B)  │ low nibbles (16B)│
└────────┴─────────────────┴──────────────────┘
```

Four bytes of high bits = 32 bits = 1 extra bit per weight.

**Formula:**

```
q5_i = low4_i | (high_bit_i << 4)      → 0…31
w_i  = d × (q5_i − 16)
```

**Bits per weight:** 4 (low) + 1 (high) + 0.5 (d) + 0.5 (high-bits packing overhead already counted) = **5.5 bits/weight**.

**Dependencies:** None.

**Comparison:** Same shape as Q4_0 but with one more bit. Q5_K is a better use of bits.

---

### Q5_1

**Summary:** Asymmetric Q5 — like Q5_0, but with a block minimum `m`.

**Block layout** — 24 bytes, 32 weights:

```
┌────────┬────────┬─────────────────┬──────────────────┐
│ d (2B) │ m (2B) │ high bits (4B)  │ low nibbles (16B)│
└────────┴────────┴─────────────────┴──────────────────┘
```

**Formula:**

```
w_i = d × q5_i + m
```

**Bits per weight:** **6.0 bits/weight**.

**Dependencies:** None.

**Comparison:** Rarely used in practice. Q5_K_M dominates this bit-range.

---

### Q8_0

**Summary:** 8-bit symmetric quant. Basically "convert every weight to a signed byte." Minimal information loss; high quality but bulky.

**Block layout** — 34 bytes, 32 weights:

```
┌────────┬──────────────────────┐
│ d (2B) │ qs: int8 × 32 (32B)  │
└────────┴──────────────────────┘
```

**Formula:**

```
w_i = d × qs_i    (qs is signed −128…+127)
```

**Bits per weight:** 8 (qs) + 0.5 (d) = **8.5 bits/weight**.

**Dependencies:** None.

**Comparison:** Near-lossless relative to F16 but roughly half the size. Commonly used as a "high-fidelity" baseline or for weights too sensitive to drop lower (token embeddings, output heads).

---

### Q8_1

**Summary:** Q8_0 plus a precomputed block sum, useful for accelerating dot products. Rarely used as a storage format; mostly an intermediate on GPU paths.

**Block layout** — 36 bytes, 32 weights:

```
┌────────┬──────────┬──────────────────────┐
│ d (2B) │ sum (2B) │ qs: int8 × 32 (32B)  │
└────────┴──────────┴──────────────────────┘
```

**Formula:** Same as Q8_0; `sum` is only used by optimized kernels, not the reconstruction.

**Bits per weight:** **9.0 bits/weight**.

**Dependencies:** None.

**Comparison:** You'll almost never see Q8_1 as the persistent storage type of a model. Q8_0 is the conventional 8-bit choice.

---

## K-Quants (256 weights/block)

The workhorse family. Each block covers **256 weights**, organised into **sub-blocks** (8 or 16 depending on type). Two scale levels — a float16 **super-block scale** `d` that multiplies everything, and a smaller (4–8 bit) **per-sub-block scale** `sc_j`. This lets K-quants track local variation much better than simple Q formats at similar bit counts.

### Q2_K

**Summary:** 2-bit K-quant. Four values per weight (00, 01, 10, 11) after scaling. 16 sub-blocks of 16 weights each, each with its own tiny scale and minimum.

**Block layout** — 84 bytes, 256 weights:

```
┌────────────────────┬──────────────────────┬────────┬──────────┐
│ sub-scales (16B)   │ 2-bit quants (64B)   │ d (2B) │ dmin (2B)│
└────────────────────┴──────────────────────┴────────┴──────────┘
```

- 16 sub-blocks of 16 weights.
- Each sub-scale byte packs a 4-bit scale + 4-bit min.
- 64 bytes × 4 weights/byte = 256 weights (2 bits each).

**Formula:**

```
w_i = d × sc_j × q_i − dmin × m_j
```

- `q_i` is 0–3.
- `sc_j`, `m_j` are 4-bit unsigned.

**Bits per weight:**

| Source | Bits/weight |
|---|---|
| 2-bit quants | 2.00 |
| sub-scales (4+4 bits × 16 sub-blocks / 256) | 0.50 |
| d + dmin (32 bits / 256) | 0.125 |
| **Total** | **~2.6 bits/weight** |

**Dependencies:** None required, but an imatrix helps at this bit level.

**Composition in a `Q2_K` file preset:**

| Tensor role | Type |
|---|---|
| most linear layers | Q2_K |
| `output.weight` | Q6_K |
| `attn_v`, first few `ffn_down` | Q4_K |
| `token_embd` | Q2_K |

**Comparison:**
- **Q2_K vs IQ2_XXS** — IQ2_XXS is smaller (~2.06 bits/weight vs 2.6) but more fragile. Q2_K is usually higher quality at the cost of ~25% more size.

---

### Q3_K

**Summary:** 3-bit K-quant. Uses the same "split low + high bits" trick as Q5_0: 2 bits in the main quants field plus a 1-bit high-mask to reach 3 bits per weight. 16 sub-blocks of 16 weights, 6-bit scales packed.

**Block layout** — 110 bytes, 256 weights:

```
┌───────────────────┬──────────────────────┬──────────────────┬────────┐
│ high-bit mask     │ 2-bit quants (64B)   │ sub-scales (12B) │ d (2B) │
│ (32B)             │                      │                  │        │
└───────────────────┴──────────────────────┴──────────────────┴────────┘
```

- 32-byte high-bit mask = 1 bit per weight.
- 12 bytes for 16 sub-scales at 6 bits each.

**Formula:**

```
q3_i = low2_i | (high_bit_i << 2)       → 0…7
w_i  = d × sc_j × (q3_i − 4)
```

**Bits per weight:**

| Source | Bits |
|---|---|
| 2-bit low quants | 2.0 |
| high-bit mask | 1.0 |
| sub-scales (6 bits × 16 / 256) | 0.375 |
| super-scale `d` | 0.0625 |
| **Total** | **~3.44 bits/weight** |

**Dependencies:** Imatrix strongly recommended at this bit level.

**Composition in `Q3_K_M` / `Q3_K_L` presets:**

| Tensor role | `Q3_K_S` | `Q3_K_M` | `Q3_K_L` |
|---|---|---|---|
| most linear | Q3_K | Q3_K | Q3_K |
| `attn_v` | Q3_K | Q4_K | Q5_K |
| `ffn_down` (first ~⅛ layers) | Q3_K | Q4_K | Q5_K |
| `output.weight` | Q5_K | Q5_K | Q5_K |

**Comparison:**
- **Q3_K vs IQ3_S** — similar bit count (~3.4). Q3_K is symmetric and linear; IQ3_S uses a codebook + sign bits. IQ3_S is typically higher quality with imatrix, comparable or slightly lower without.

---

### Q4_K

**Summary:** 4-bit K-quant. 8 sub-blocks of 32 weights, each with its own 6-bit scale + 6-bit minimum (asymmetric). The current sweet-spot for "4-bit quality."

**Block layout** — 144 bytes, 256 weights:

```
┌────────┬──────────┬─────────────────┬──────────────┬──────────────┬──────────────────────┐
│ d (2B) │ dmin (2B)│ sub-scales (4B) │ sub-mins (4B)│ overflow (4B)│ 4-bit quants (128B)  │
└────────┴──────────┴─────────────────┴──────────────┴──────────────┴──────────────────────┘
```

The 12-byte packed section is 8 scales + 8 mins at 6 bits each = 96 bits = 12 bytes. Split into `sub-scales (4B)`, `sub-mins (4B)`, and `overflow (4B)` because 6-bit values don't align cleanly to byte boundaries.

**Formula:**

```
w_i = d × sc_j × q_i − dmin × m_j
```

- `q_i` is a 4-bit nibble, 0–15.
- `sc_j`, `m_j` are 6-bit sub-scale / sub-min.

**Bits per weight:**

| Source | Bits |
|---|---|
| nibbles | 4.00 |
| sub-scales + mins (12 bytes / 256) | 0.375 |
| d + dmin (32 bits / 256) | 0.125 |
| **Total** | **~4.5 bits/weight** |

**Dependencies:** Imatrix helpful but not required.

**Composition in `Q4_K_M` / `Q4_K_S` presets:**

| Tensor role | `Q4_K_S` | `Q4_K_M` |
|---|---|---|
| most linear | Q4_K | Q4_K |
| `attn_v` | Q4_K | Q6_K |
| `ffn_down` (first ~⅛ layers) | Q4_K | Q6_K |
| `output.weight` | Q6_K | Q6_K |

**Comparison:**
- **Q4_K vs Q4_0** — same nominal bit count, but Q4_K has per-sub-block scales. Much better quality at similar size. Always prefer Q4_K.
- **Q4_K vs IQ4_NL** — very similar sizes. Q4_K is faster on most CPUs; IQ4_NL can be slightly higher quality on some tensors. Q4_K is safer default.
- **Q4_K_M vs Q5_K_S** — Q5_K_S is ~15% larger. Perplexity-wise, the two are often surprisingly close.

---

### Q5_K

**Summary:** Like Q4_K, plus one extra bit per weight sitting in a separate 32-byte high-bit mask.

**Block layout** — 176 bytes, 256 weights:

```
┌────────┬──────────┬──────────────────┬──────────────────┬──────────────────────┐
│ d (2B) │ dmin (2B)│ sub-scales (12B) │ high bits (32B)  │ 4-bit quants (128B)  │
└────────┴──────────┴──────────────────┴──────────────────┴──────────────────────┘
```

**Formula:**

```
q5_i = low4_i | (high_bit_i << 4)      → 0…31
w_i  = d × sc_j × q5_i − dmin × m_j
```

**Bits per weight:** ~**5.5 bits/weight**.

**Dependencies:** Imatrix optional.

**Composition in `Q5_K_M` / `Q5_K_S`:**

| Tensor role | `Q5_K_S` | `Q5_K_M` |
|---|---|---|
| most linear | Q5_K | Q5_K |
| `attn_v` | Q5_K | Q6_K |
| `ffn_down` (first ~⅛ layers) | Q5_K | Q6_K |
| `output.weight` | Q6_K | Q6_K |

**Comparison:**
- **Q5_K vs Q4_K_M** — ~20% more bytes, quality gain usually small but consistent.
- **Q5_K_M vs Q6_K** — Q6_K is another ~15% bigger for another small bump.

---

### Q6_K

**Summary:** 6-bit K-quant. Splits each weight into low-4 + high-2 stored separately. 16 sub-blocks of 16 weights, each with a straight int8 scale. Near-lossless for most purposes.

**Block layout** — 210 bytes, 256 weights:

```
┌──────────────────────┬──────────────────────┬───────────────────┬────────┐
│ low 4-bit quants     │ high 2-bit quants    │ sub-scales (16B)  │ d (2B) │
│ (128B)               │ (64B)                │ (int8 × 16)       │        │
└──────────────────────┴──────────────────────┴───────────────────┴────────┘
```

**Formula:**

```
q6_i = low4_i | (high2_i << 4)      → 0…63
w_i  = d × sc_j × (q6_i − 32)
```

**Bits per weight:** 4 + 2 + 0.5 (sub-scales) + 0.0625 (d) = **~6.5 bits/weight**.

**Dependencies:** None.

**Comparison:**
- **Q6_K vs Q5_K_M** — ~15% larger, marginally better quality. Often the smallest "set and forget" quant for quality-critical work.
- **Q6_K vs Q8_0** — Q6_K is ~25% smaller; Q8_0 is essentially lossless. Q6_K is usually good enough.

---

### Q8_K

**Summary:** 8-bit K-quant — but you almost never see this as a persistent storage type. It's used internally as an **intermediate** dot-product format when multiplying a Q-quant weight by an activation. It also carries precomputed 16-bit "block sums" to accelerate matrix multiplies.

**Block layout** — 292 bytes, 256 weights:

```
┌────────┬──────────────────────────┬───────────────────────────┐
│ d: f32 │ qs: int8 × 256 (256B)    │ bsums: int16 × 16 (32B)   │
│ (4B!)  │                          │                           │
└────────┴──────────────────────────┴───────────────────────────┘
```

Note: `d` is **float32** (4 bytes), not float16 — because this format is about numerical accuracy during dot products, not compactness.

**Formula:**

```
w_i = d × qs_i
```

**Bits per weight:** ~**9.1 bits/weight**.

**Dependencies:** None.

**Comparison:** Mostly an internal format. If you see it as a storage type, it's rarely worth vs Q6_K (smaller) or F16 (similar size).

---

## IQ-Quants (importance-matrix, 256 weights/block)

The IQ family uses a **hardcoded codebook** baked into ggml's C source. Each "quant" in the file is an **index** into a table of pre-chosen small weight vectors. This lets IQ types go below 4 bits/weight while staying usable. Best paired with an **imatrix** (see [§ The Imatrix](#the-imatrix)).

The codebook exists because trained neural-net weights have predictable statistical structure. The grid was built by clustering real model weights into the most common small "shapes." The grid ships with ggml and is identical for every IQ file ever produced — if you change it, all existing files decode to garbage. So it's frozen.

### IQ1_S

**Summary:** Nominally 1-bit-per-weight quantisation — but it squeaks up to ~1.56 bits via sub-block extras. Extreme compression; only usable on large models where even brutal accuracy loss is survivable.

**Block layout** — 50 bytes, 256 weights:

```
┌────────┬─────────────────────────┬──────────────────┐
│ d (2B) │ codebook indices (32B)  │ high bits (16B)  │
└────────┴─────────────────────────┴──────────────────┘
```

**Formula:**

```
w_i = d × LUT[idx_i, qh_i]
```

The codebook effectively encodes several possible ±1/0 patterns per 8-weight group; `qh` selects among variants and also carries sub-block scale info.

**Bits per weight:** ~**1.56 bits/weight**.

**Dependencies:** **Imatrix strongly required** — quality collapses without one.

**Comparison:**
- **IQ1_S vs IQ1_M** — IQ1_M has a denser high-bit field and packed per-sub-block scales, gaining ~0.2 bits/weight and measurably better quality.
- **IQ1_S vs IQ2_XXS** — IQ2_XXS is ~0.5 bit/weight bigger but substantially more accurate. Use IQ1 only when IQ2 is still too big.

---

### IQ1_M

**Summary:** IQ1_S's beefier cousin. Trades size for dropping the per-block `d` field in favour of scales packed *across* the qh and scales arrays. Effectively ~1.75 bits/weight.

**Block layout** — 56 bytes, 256 weights:

```
┌─────────────────────────┬──────────────────┬──────────────────┐
│ codebook indices (32B)  │ high bits (16B)  │ packed scales(8B)│
└─────────────────────────┴──────────────────┴──────────────────┘
```

**No separate `d` field** — the super-block scale is encoded inside the packed scales field.

**Formula:**

```
w_i = packed_scale × LUT[idx_i, qh_i]
```

**Bits per weight:** ~**1.75 bits/weight**.

**Dependencies:** Imatrix strongly required.

**Comparison:** See IQ1_S above. IQ1_M is slightly larger and more accurate; otherwise analogous.

---

### IQ2_XXS

**Summary:** The smallest "real" IQ2 — ~2 bits per weight, minimum metadata. Codebook-only: 8-bit index per group of 4 weights, plus a single float16 scale per 256 weights.

**Block layout** — 66 bytes, 256 weights:

```
┌────────┬─────────────────────────┐
│ d (2B) │ codebook indices (64B)  │
└────────┴─────────────────────────┘
```

**Formula:**

```
w_i = d × LUT[idx_group]
```

**Bits per weight:** ~**2.06 bits/weight**.

**Dependencies:** **Imatrix required** for reasonable quality.

**Comparison:**
- **IQ2_XXS vs IQ2_XS** — IQ2_XS adds an 8-byte sub-scales field (~0.25 bit/weight more) and is noticeably better.
- **IQ2_XXS vs Q2_K** — IQ2_XXS is ~20% smaller. With imatrix it's competitive; without, Q2_K wins.

---

### IQ2_XS

**Summary:** IQ2_XXS with per-sub-block scales. Fills the gap between XXS and S.

**Block layout** — 74 bytes, 256 weights:

```
┌────────┬─────────────────────────┬──────────────────┐
│ d (2B) │ codebook indices (64B)  │ sub-scales (8B)  │
└────────┴─────────────────────────┴──────────────────┘
```

**Formula:**

```
w_i = d × sc_j × LUT[idx_group]
```

**Bits per weight:** ~**2.31 bits/weight**.

**Dependencies:** Imatrix strongly recommended.

**Comparison:** See IQ2_XXS / IQ2_S.

---

### IQ2_S

**Summary:** IQ2 with both sub-scales and an extra "high bits" field that extends each codebook index by 1 bit (so 256 → 512 entries). More coverage of weight patterns.

**Block layout** — 82 bytes, 256 weights:

```
┌────────┬─────────────────────────┬──────────────────┬──────────────────┐
│ d (2B) │ codebook indices (64B)  │ high bits (8B)   │ sub-scales (8B)  │
└────────┴─────────────────────────┴──────────────────┴──────────────────┘
```

**Formula:**

```
w_i = d × sc_j × LUT[idx_i, qh_i]
```

**Bits per weight:** ~**2.56 bits/weight**.

**Dependencies:** Imatrix strongly recommended.

**Comparison:**
- **IQ2_S vs Q2_K** — comparable size; IQ2_S usually wins with imatrix. Without imatrix, Q2_K is safer.
- **IQ2_S vs IQ3_XXS** — IQ3_XXS is ~0.5 bit bigger but a step up in quality.

---

### IQ3_XXS

**Summary:** 3-bit IQ with signs packed into the quants field. Smallest "real" IQ3 variant.

**Block layout** — 98 bytes, 256 weights:

```
┌────────┬─────────────────────────────────────┐
│ d (2B) │ packed 3-bit indices + signs (96B)  │
└────────┴─────────────────────────────────────┘
```

Signs and indices are interleaved in the 96-byte field — each group of 4 weights uses ~12 bits for the index + 7 sign bits + some sub-scale info packed in.

**Formula:**

```
w_i = d × sign_i × LUT[idx_group]
```

**Bits per weight:** ~**3.06 bits/weight**.

**Dependencies:** Imatrix strongly recommended.

**Comparison:**
- **IQ3_XXS vs IQ3_S** — IQ3_S is ~0.4 bit/weight bigger (explicit signs + sub-scales + 9-bit index). Notably better quality.

---

### IQ3_S

**Summary:** The "correct" 3-bit IQ. Explicit 1-bit sign per weight, 9-bit codebook index per group of 4, and 4-bit sub-scales per 32-weight sub-block. The canonical example for walking through IQ architecture.

**Block layout** — 110 bytes, 256 weights:

```
┌────────┬─────────────────────┬──────────────────┬─────────────────┬──────────────────┐
│ d (2B) │ 3-bit indices (64B) │ high bits (8B)   │ sign bits (32B) │ sub-scales (4B)  │
└────────┴─────────────────────┴──────────────────┴─────────────────┴──────────────────┘
```

- `qs` (64B) = 1 byte per group of 4 weights (low 8 bits of a 9-bit codebook index).
- `qh` (8B) = 1 extra high bit per group → 9-bit index into a 512-entry grid.
- `signs` (32B) = 1 sign bit per individual weight.
- `scales` (4B) = 4-bit scale per sub-block, packed two per byte (8 sub-blocks × 4 bits = 32 bits).

**Formula:**

```
index     = qs[g] | (qh_bit << 8)
magnitude = LUT_group = iq3s_grid[index][p]      // p = position in group, 0..3
sub_scale = 1 + 2 × scales[j]                    // odd integer 1, 3, 5, …, 31
w_i       = d × sub_scale × sign_i × magnitude
```

**Bits per weight:**

| Source | Bits |
|---|---|
| 9-bit index / 4 weights | 2.25 |
| sign bits | 1.00 |
| sub-scales (4 bits × 8 / 256) | 0.125 |
| super-scale `d` | 0.0625 |
| **Total** | **~3.44 bits/weight** |

**Dependencies:** Imatrix strongly recommended.

**Why signs are separate.** Storing signs in the codebook itself would require 16× as many entries (4 weights × 2 signs each) for the same expressive power. The separate 1-bit-per-weight layout keeps the codebook small enough to fit in L1 cache, which is essential for fast dequantization.

**Comparison:** See IQ3_XXS above, and `IQ3_M` preset in [File-level Presets](#file-level-presets-llamacpp-recipes).

---

### IQ4_NL

**Summary:** "Non-Linear" 4-bit quant. Same block shape as Q4_0 (18 bytes / 32 weights), but the 4-bit nibbles index into a hardcoded **16-entry non-linear table** of float values instead of using a linear `(nibble − 8)` mapping.

The NL table is hand-tuned to match typical trained-weight distributions, which tend to have more mass near zero and long tails — a non-linear mapping represents this much better than equal-spaced levels.

**Block layout** — 18 bytes, 32 weights:

```
┌────────┬─────────────────┐
│ d (2B) │ nibbles (16B)   │
└────────┴─────────────────┘
```

**Formula:**

```
w_i = d × NL[nibble_i]

// NL is a 16-entry float table, hardcoded in ggml:
// approximately [-127/127, -104/127, …, 0, …, 104/127, 127/127]
// but non-linearly spaced
```

**Bits per weight:** **4.5 bits/weight** (same as Q4_0).

**Dependencies:** Imatrix recommended; not strictly required.

**Comparison:**
- **IQ4_NL vs Q4_0** — same size, better quality.
- **IQ4_NL vs Q4_K** — comparable quality; Q4_K is slightly more flexible per sub-block, IQ4_NL is slightly faster on some kernels.
- **IQ4_NL vs IQ4_XS** — IQ4_XS is smaller (~4.25 bits/weight) by using 256-weight super-blocks with sub-scales.

---

### IQ4_XS

**Summary:** IQ4_NL applied to 256-weight super-blocks, with 6-bit sub-scales per 32-weight sub-block. Smaller and more accurate than IQ4_NL.

**Block layout** — 136 bytes, 256 weights:

```
┌────────┬──────────────────┬──────────────────────┐
│ d (2B) │ sub-scales (6B)  │ nibbles (128B)       │
└────────┴──────────────────┴──────────────────────┘
```

**Formula:**

```
w_i = d × sc_j × NL[nibble_i]
```

**Bits per weight:** ~**4.25 bits/weight**.

**Dependencies:** Imatrix recommended.

**Comparison:**
- **IQ4_XS vs Q4_K_M** — IQ4_XS is smaller (~0.25 bit/weight less). Quality is comparable; IQ4_XS wins with a good imatrix, Q4_K_M wins without.
- **IQ4_XS vs IQ4_NL** — 256-weight blocks with sub-scales vs 32-weight blocks. XS is more size-efficient and slightly more accurate.

---

## File-level Presets (llama.cpp recipes)

When you see a file named e.g. `google_gemma-3-270m-it-IQ3_M.gguf`, the `IQ3_M` isn't a tensor type — it's a **llama.cpp quantization recipe** describing which tensor type to apply to which tensor role. Every file is a mix.

The recipes live in [`llama.cpp`'s quantize code](https://github.com/ggerganov/llama.cpp/blob/master/src/llama-quant.cpp). The general shape:

```
for each tensor:
  decide its role from its name (attn_v, ffn_down, output, token_embd, …)
  decide its layer index
  look up in the recipe table → pick a specific quant type
```

### Suffix conventions on presets

- `_S` (small) — stays on the base type almost everywhere. Smallest, lowest quality.
- `_M` (medium) — promotes certain tensor roles to the next tier. Best-quality-per-byte sweet spot.
- `_L` (large) — even more aggressive promotion. Closer to the next whole-bit tier.

### Tensor roles that get special treatment

These are the tensors the quantizer consistently treats as "sensitive":

| Role | Why it's sensitive |
|---|---|
| `output.weight` | The unembedding matrix — directly gates logits. Errors here fall straight into token probabilities. |
| `token_embd.weight` | The embedding table. Heavily shared; errors appear everywhere. |
| `attn_v.weight` | Attention value projections. Attention is sensitive to rounding. |
| `ffn_down.weight` | The "down-projection" at the end of each FFN. Accumulates errors. |

### IQ3 family recipe

| Tensor role | `IQ3_XXS` | `IQ3_S` | `IQ3_M` |
|---|---|---|---|
| most linear layers | IQ3_XXS | IQ3_S | IQ3_S |
| `attn_v.weight` | IQ3_S | Q4_K | Q4_K |
| `ffn_down` (first ~⅛ layers) | Q3_K / IQ3_S | Q4_K | Q4_K |
| `ffn_down` (other layers) | IQ3_XXS | IQ3_S | **IQ4_XS** |
| `output.weight` | Q5_K | Q5_K | Q5_K |
| `token_embd.weight` | Q2_K / IQ3_S | IQ3_S | IQ3_S |

So `IQ3_S` and `IQ3_M` share the same *base* (IQ3_S); the difference is that `IQ3_M` additionally promotes **all `ffn_down` tensors** to IQ4_XS. That one line accounts for the whole quality delta between them.

### IQ2 family recipe

| Tensor role | `IQ2_XXS` | `IQ2_XS` | `IQ2_S` | `IQ2_M` |
|---|---|---|---|---|
| most linear | IQ2_XXS | IQ2_XS | IQ2_S | IQ2_S |
| `attn_v` | IQ2_XS | IQ2_S | Q4_K | Q4_K |
| `ffn_down` | IQ2_XXS | IQ2_XS | IQ2_S | IQ4_XS |
| `output.weight` | Q4_K | Q4_K | Q4_K | Q4_K |
| `token_embd` | Q2_K | Q2_K | IQ2_S | IQ2_S |

### IQ1 family recipe

| Tensor role | `IQ1_S` | `IQ1_M` |
|---|---|---|
| most linear | IQ1_S | IQ1_M |
| `attn_v` | IQ2_XXS | IQ2_S |
| `ffn_down` | IQ2_XXS | IQ2_S |
| `output.weight` | Q5_K | Q5_K |
| `token_embd` | IQ2_XXS | IQ2_S |

### Q3_K family recipe

| Tensor role | `Q3_K_S` | `Q3_K_M` | `Q3_K_L` |
|---|---|---|---|
| most linear | Q3_K | Q3_K | Q3_K |
| `attn_v` | Q3_K | Q4_K | Q5_K |
| `ffn_down` (first ~⅛ layers) | Q3_K | Q4_K | Q5_K |
| `output.weight` | Q5_K | Q5_K | Q5_K |

### Q4_K family recipe

| Tensor role | `Q4_K_S` | `Q4_K_M` |
|---|---|---|
| most linear | Q4_K | Q4_K |
| `attn_v` | Q4_K | Q6_K |
| `ffn_down` (first ~⅛ layers) | Q4_K | Q6_K |
| `output.weight` | Q6_K | Q6_K |

### Q5_K family recipe

| Tensor role | `Q5_K_S` | `Q5_K_M` |
|---|---|---|
| most linear | Q5_K | Q5_K |
| `attn_v` | Q5_K | Q6_K |
| `ffn_down` (first ~⅛ layers) | Q5_K | Q6_K |
| `output.weight` | Q6_K | Q6_K |

> Exact layer thresholds and tensor-name patterns evolve between llama.cpp releases. The tables above show the *common* pattern — check `llama-quant.cpp` for the definitive current rules.

---

## The Imatrix

An **importance matrix** (imatrix) is a file produced by running a small amount of calibration text through the model at full precision and recording, for every weight, how much that weight contributed to the output. The result is a per-weight "importance score."

During quantization, the imatrix is used as an error-weighting term: the quantizer prefers codebook matches that preserve **important** weights at the cost of being sloppier on **unimportant** ones. This is what makes sub-4-bit quantization usable at all.

**When you need one:**

| Quant type | Imatrix |
|---|---|
| F32 / F16 / BF16 | N/A |
| Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 | Not needed |
| Q2_K | Strongly recommended |
| Q3_K | Strongly recommended |
| Q4_K / Q5_K / Q6_K | Optional (small gain) |
| Q8_K | Not needed |
| IQ1_S / IQ1_M | **Required** (quality collapses without it) |
| IQ2_* | **Required** |
| IQ3_* | Strongly recommended |
| IQ4_NL / IQ4_XS | Recommended |

The imatrix is consumed at quantization time and thrown away — it's **not** stored in the GGUF file. You can't tell from the file alone whether an imatrix was used, though filename conventions often include `imat` to hint.

---

## Comparison Cheatsheet

All types, sorted by bits/weight ascending.

| Type | Bits/wt | Block (B / wts) | Imatrix | Notes |
|---|---|---|---|---|
| IQ1_S | 1.56 | 50 / 256 | required | Smallest practical. Large models only. |
| IQ1_M | 1.75 | 56 / 256 | required | No separate `d`. |
| IQ2_XXS | 2.06 | 66 / 256 | required | Codebook only. |
| IQ2_XS | 2.31 | 74 / 256 | required | + sub-scales. |
| IQ2_S | 2.56 | 82 / 256 | required | + high bits. |
| Q2_K | 2.6 | 84 / 256 | recommended | Linear alternative. |
| IQ3_XXS | 3.06 | 98 / 256 | recommended | Signs packed. |
| Q3_K | 3.44 | 110 / 256 | recommended | Linear. |
| IQ3_S | 3.44 | 110 / 256 | recommended | Explicit signs + 512 grid. |
| IQ4_XS | 4.25 | 136 / 256 | recommended | NL + sub-scales. |
| IQ4_NL | 4.5 | 18 / 32 | recommended | Q4_0 shape, non-linear. |
| Q4_0 | 4.5 | 18 / 32 | — | Legacy, linear. |
| Q4_K | 4.5 | 144 / 256 | optional | Modern default. |
| Q4_1 | 5.0 | 20 / 32 | — | Legacy + min. |
| Q5_0 | 5.5 | 22 / 32 | — | Legacy. |
| Q5_K | 5.5 | 176 / 256 | optional | Good quality. |
| Q5_1 | 6.0 | 24 / 32 | — | Legacy. |
| Q6_K | 6.5 | 210 / 256 | — | Near-lossless. |
| Q8_0 | 8.5 | 34 / 32 | — | Simple, robust. |
| Q8_1 | 9.0 | 36 / 32 | — | + block sum, rare as storage. |
| Q8_K | 9.1 | 292 / 256 | — | Intermediate format, not storage. |
| BF16 | 16 | 2 / 1 | — | Wide range, reduced mantissa. |
| F16 | 16 | 2 / 1 | — | Standard half-precision. |
| F32 | 32 | 4 / 1 | — | Full precision. |

---

## Choosing a Quant

A rough decision tree for the common case of quantizing a language model for local inference:

```
Need maximum quality, have the RAM/VRAM?    → Q8_0 or F16
Want near-lossless, save ~25%?              → Q6_K
Sweet spot: quality vs size?                → Q5_K_M or Q4_K_M
Running on 8 GB, need it smaller still?     → IQ4_XS or IQ3_M (with imatrix)
Tight memory on a big model?                → IQ3_S / IQ3_XXS
Squeezing a huge model onto a laptop?       → IQ2_M or IQ2_S (imatrix required)
Absolute last resort?                       → IQ1_M (only usable on 70B+)
```

**Rules of thumb:**

- Every step down the table costs real quality. Start at Q5_K_M and only go lower if you have to.
- Below 4 bits, **always** use imatrix files when available.
- A `_M` preset at N bits is usually a better choice than an `_S` preset at N+1 bits.
- For sub-3-bit quants, model size matters: IQ2 on a 70B often beats IQ4 on a 7B. Big models have more redundancy to absorb quantization noise.
- Speed: on CPU, K-quants are generally fastest. IQ-quants have improved a lot but still have more kernel overhead.
