# tediet

Put your diffusion pipeline's text encoder on a diet.

[日本語README](README.md)

Modern image/video generation models carry huge text encoders — T5-XXL (~9 GB),
Gemma, Qwen3-VL (~16 GB) — that run **once per job** while the diffusion
transformer runs for every step. Keeping the whole encoder resident wastes most
of a 16–24 GB GPU. `tediet` provides two small, composable, **output-exact**
transformations that reclaim that memory without the latency of generic
CPU-offloading:

| Technique | What it does | VRAM effect |
|---|---|---|
| **diet** | Moves the token-embedding table and the (unused) LM head to the CPU, and skips the full-vocabulary logits computation | −2.3 GiB on Qwen3-VL, −1.9 GiB on Gemma |
| **stream** | Keeps the decoder layers in pinned host memory and copies them to fixed GPU ring-buffer slots only while an encode runs, two layers ahead of compute | Layer stack residency −12.9 GiB → ~1.3 GiB (Qwen3-VL bf16) |

Both are **bit-exact**: the hidden states are identical to the fully resident
model, because no computation changes — only where two lookups live, whether an
unused projection runs, and where the unchanged weights are stored.

## Measured

Qwen-Image 2.1 (Qwen3-VL 16 GB text encoder, bf16), RTX PRO 4000 Blackwell 24 GB,
1024×1024, end-to-end per image:

| Configuration | Text-encoder residency | e2e time |
|---|---|---|
| Fully resident | 16.3 GiB | baseline |
| diet + stream (window 2) | **~1.5 GiB** | **+0.6 s** |

Against diffusers' own `apply_group_offloading` on the same Qwen3-VL
(reproducible with `benchmarks/bench_text_encoder.py`; 27-token prompt,
encode only, mean of 5):

| Implementation | Resident | Encode | Peak |
|---|---|---|---|
| Fully resident | 16.34 GiB | 0.040 s | 16.39 GiB |
| **tediet (diet + stream, window 2)** | **2.19 GiB** | **0.886 s** | **2.20 GiB** |
| stock leaf_level + `use_stream=True` | 2.28 GiB | 1.032 s | 3.64 GiB |
| stock leaf_level, no stream | 2.28 GiB | 6.959 s | 3.44 GiB |

The gap against the stock implementation is 1.2–8x depending on its
configuration. To be fair: tuned all the way to leaf_level + stream, the
stock implementation comes close. tediet is still ~17% faster to encode,
peaks 40% lower (the fixed ring buffer allocates nothing), and has none of
the traps of the naive setups (block_level on the whole encoder onloads the
entire base model as one group; block_level on inner submodules crashes on
the unhooked embedding). On LTX-2.5's Gemma NF4 encoder we measured a 3–6x
gap against the stock implementation.

The gap comes from three design choices: hooks attach per layer instead of per
offload group, restoring a layer is pointer reassignment (no device-to-host
copy — the weights never change), and the ring buffer makes an encode allocate
**zero** new GPU memory, which also keeps the allocator stable next to CUDA
Graph memory pools.

## Usage

```python
from tediet import apply_diet, apply_stream

# Qwen3-VL (Qwen-Image 2.x pipelines)
freed = apply_diet(pipe.text_encoder, embed_path="model.language_model.embed_tokens")
pinned = apply_stream(pipe.text_encoder, layers_path="model.language_model.layers",
                      device="cuda:0", window=2)
```

The two calls compose in either order, and each is idempotent. `apply_diet`
assumes the pipeline reads `outputs.hidden_states` (every diffusers text-encoder
path does); `apply_stream` keeps one ring of slots per layer signature, so stacks with
alternating layer kinds work too.

### Model recipes

| Model (pipeline) | `embed_path` | `layers_path` |
|---|---|---|
| Qwen3-VL (Qwen-Image 2.x) | `model.language_model.embed_tokens` | `model.language_model.layers` |
| Gemma (LTX-2.x) | `model.language_model.embed_tokens` | `model.language_model.layers` |
| T5-XXL (FLUX, SD3) | `encoder.embed_tokens`¹ | `encoder.block` |

¹ T5 is encoder-only: pass `lm_head_path=None` and skip `apply_lm_head_skip`.

All three recipes are validated bit-identical on real hardware
(`benchmarks/bench_text_encoder.py`):

| Model | Resident (before → after) | Encode (before → after) |
|---|---|---|
| Qwen3-VL bf16 (Qwen-Image 2.1) | 16.34 → 2.19 GiB | 0.040 → 0.886 s |
| Gemma NF4 (LTX-2.5) | 7.63 → 0.80 GiB | 0.069 → 0.405 s |
| T5-XXL bf16 (encoder-only) | 8.87 → 1.47 GiB | 0.023 → 0.570 s |

On Gemma NF4, stock group offloading crashes with device mismatches at both
leaf_level and block_level, while tediet runs as-is. Conversely, on a plain
T5 encoder the stock leaf_level + stream configuration is a close competitor
(0.633 s, with lower residency than tediet). Models with alternating
layer kinds (Gemma's sliding/global attention layers) are handled by keeping
one ring of slots per layer signature.

Recipes for more models, and the design rationale, live in [docs/](docs/).

## Requirements and constraints

- The model must stay **resident** otherwise: do not combine with
  `enable_model_cpu_offload` / `enable_sequential_cpu_offload` / accelerate
  hooks, which move whole components with `.to()` and fight the CPU placement.
- `stream` pins host RAM equal to the layer stack (12.9 GB for Qwen3-VL bf16).
- Supported layer parameters: plain fp32/bf16/fp16 tensors and bitsandbytes
  4-bit (`quant_state` tensors travel with the layer). TorchAO tensor
  subclasses are untested.
- Encodes must not run concurrently from multiple threads.

## License

MIT.
