# Design notes: why this is faster than stock group offloading

[日本語版](design.md)

This document explains the design rationale behind tediet's two
transformations (diet / stream) and why they run at 3–6x lower overhead than
diffusers' own `apply_group_offloading`.

## The observations everything follows from

Three structural facts hold for the text encoder of a diffusion pipeline:

1. **The text encoder runs once per job.** The diffusion transformer runs for
   6–50 steps; the encoder runs once at the start. A fully resident setup
   nevertheless keeps the encoder in VRAM the whole time.
2. **The pipeline reads only `hidden_states` from the encoder output.**
   Pipelines that reuse a `*ForConditionalGeneration` model compute the LM
   head (a projection onto the full vocabulary) on every encode and throw the
   logits away.
3. **The encoder weights never change during inference.** Nothing ever needs
   to be written back to the host.

Every design decision in tediet follows from these three facts.

## diet: the embedding bridge and the LM-head skip

### CPU-bridged embedding table (`apply_embed_bridge`)

The token embedding table (for Qwen3-VL: 151936×4096 bf16 = 1.16 GiB) is used
for exactly one gather at the start of an encode. tediet moves the module to
the CPU and replaces its `forward` with a bridge: the input ids travel to the
CPU, the gather happens there, and the result travels back to the caller's
device. A 1024-token round trip is a few megabytes; the measured cost is
negligible.

**Why not pass `inputs_embeds` instead:** the naive alternative — computing
the embeddings in the pipeline and passing `inputs_embeds` — is rejected
deliberately. Multimodal transformers models contain code paths (for example
`get_placeholder_mask()`) that locate placeholder positions by comparing
against the special tokens' embedding vectors, and those paths reference the
embedding module itself. With the module on the CPU they fail with a device
mismatch, and the mask-derivation branch itself diverges from the production
path. Because tediet swaps the behavior **at module level**, every internal
caller — the ordinary lookup and the special-token comparison alike — keeps
working unchanged.

### LM-head skip (`apply_lm_head_skip`)

tediet replaces the encoder's `forward` with a thin wrapper that calls the
inner base model with the same arguments and returns its output as-is. The
base model is what produces `hidden_states`, so the returned values are built
by exactly the same code path. This removes:

- the transient full-vocabulary logits tensor (about 0.3 GiB at 1024 tokens);
- the head itself from the GPU (1.16 GiB on Qwen3-VL), when the head is
  **untied** from the embeddings. The tie check compares parameter
  `data_ptr`s, so it is automatic. On tied models the embedding bridge
  already moved the shared weight to the CPU and no extra step is needed.

## stream: ring-buffered layer streaming

### Mechanism

All tensors of the decoder layers (for Qwen3-VL: 36 layers, 12.9 GiB bf16)
move to pinned host memory. While an encode runs, a dedicated CUDA copy
stream prefetches layers `window` ahead of compute (default 2). The GPU-side
destination is a set of `window + 1` **fixed slots**, reused as a ring.

The synchronization contract takes three lines:

- The copy stream waits on `slot_free[slot]` — the event recorded when the
  layer that last used this slot finished computing — before overwriting it.
- The compute stream waits on `copied[i]` — the event recorded by the copy
  stream — before running layer `i`.
- `slots = window + 1` guarantees the slot being written is never the slot
  the current layer is computing from.

With a window of 2 on PCIe 4.0 x16, the per-layer transfer (measured ~2.4 ms)
hides completely behind the per-layer compute (~4.4 ms). Widening the window
to 4 or 8 changes nothing.

### Why it beats stock `apply_group_offloading`

Against the official implementation of the same concept (pinned + stream +
prefetch), measured on LTX-2.5's Gemma NF4 encoder:

| Implementation | Encode time |
|---|---|
| Fully resident | 0.213 s |
| tediet stream (window 2) | 0.369 s |
| stock group offloading, `use_stream=True` | 1.25 s |
| stock group offloading, no stream | 2.2 s |

The gap comes from three design decisions:

1. **Hook granularity.** The stock implementation forces
   `num_blocks_per_group=1` when `use_stream=True`, and its per-group
   bookkeeping hook (~22 ms × 48 groups measured) becomes the dominant cost.
   tediet attaches pre/post hooks directly to the layers, and the hooks do
   nothing but event operations and pointer assignment.
2. **Restoring is pointer reassignment.** The weights never change during
   inference, so "unloading" a layer means pointing each parameter's `data`
   back at its pinned CPU tensor. No device-to-host copy ever happens.
3. **Zero allocation (the ring buffer).** The stock implementation — and an
   earlier version of this one — allocated fresh GPU tensors per layer with
   `.to(device)`. Switching to `copy_(non_blocking=True)` into fixed slots
   makes an encode allocate zero new GPU memory.

### The incident that motivated the ring buffer

The earlier allocate-per-layer version (with `record_stream` deferring
reuse) worked fine on its own under the expandable_segments allocator. Placed
in the same process as a CUDA Graph memory pool, however, expandable_segments
stopped applying, cross-stream allocate/free pairs were never reused, and the
reservations piled up to the full layer stack (~14 GiB) and OOMed. The fixed
ring buffer bypasses the allocator entirely, so this failure mode cannot
occur — and even without CUDA Graphs, taking the allocator out of the encode
path is the right call.

## Validation discipline: bit-exactness

tediet does not accept "faster, but the image changed slightly". Neither
transformation changes any computation, so the hidden states must be
**bit-identical** before and after — and this is verified per target:

- Gemma (NF4, LTX-2.5): identical; two consecutive runs also validate the
  restore path.
- Qwen3-VL (bf16, Qwen-Image 2.1): identical, same two-pass protocol
  (`tests/test_equivalence_qwen3vl.py`).

When porting to a new model, record a fully resident reference output first
and compare with `torch.equal` after applying tediet. A mismatch means the
model lacks one of the structural assumptions below.

## Porting to a new model

Porting requires identifying three module paths:

1. `embed_path`: the token embedding module (an `nn.Embedding`-like).
2. `layers_path`: the `ModuleList` of structurally identical layers.
3. `inner_path` / `lm_head_path`: for `*ForConditionalGeneration`-style
   encoders, the inner base model and the head. For encoder-only models
   (T5 family), pass `lm_head_path=None` and skip `apply_lm_head_skip`.

Constraints: resident-only setups (no accelerate-style offloading), pinned
host RAM equal to the layer stack, no concurrent encodes. Models whose layers
are not structurally identical (mixed block types) fail the ring buffer's
assertion.
