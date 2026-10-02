"""Text-encoder "diet": move the token embedding table and the LM head off the GPU.

Diffusion pipelines call their text encoder once per job and read only
``outputs.hidden_states``. Two large components of a causal-LM text encoder are
therefore dead weight on the GPU:

- ``embed_tokens``: the embedding table is read once per encode via a gather.
  ``apply_embed_bridge`` moves the module to CPU and replaces its ``forward``
  with a bridge that ships the (tiny) input ids to the CPU, gathers there, and
  ships the (small) result back. The module object itself is unchanged, so
  every internal caller — including code paths that compare against special
  token embedding vectors — keeps working.
- ``lm_head``: generation pipelines never read logits. ``apply_lm_head_skip``
  replaces the encoder's ``forward`` with a thin wrapper that calls the inner
  base model directly, so the full-vocabulary logits tensor is never
  materialized, and moves the (untied) head to the CPU.

Both transformations are output-exact: the hidden states are bit-identical to
the untouched model, because no computation is changed — only where two
lookups live and whether an unused projection runs.

Constraint: these helpers assume the model stays resident (no
``enable_model_cpu_offload``-style hooks). Accelerate hooks move whole
components back and forth with ``.to()`` and would fight the CPU placement.
"""
from __future__ import annotations

import torch


def _resolve(root: torch.nn.Module, path: str) -> torch.nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def apply_embed_bridge(model: torch.nn.Module, embed_path: str) -> float:
    """Move the embedding module at ``embed_path`` to CPU behind a device bridge.

    Returns the GPU memory freed, in GiB. Idempotent: a second call returns 0.
    """
    embed = _resolve(model, embed_path)
    if getattr(embed, "_tediet_bridged", False):
        return 0.0

    freed = sum(p.numel() * p.element_size() for p in embed.parameters())
    orig_forward = embed.forward

    def bridged_forward(input_ids: torch.Tensor) -> torch.Tensor:
        return orig_forward(input_ids.to("cpu")).to(input_ids.device)

    embed.to("cpu")
    embed.forward = bridged_forward
    embed._tediet_bridged = True
    return freed / 1024**3


def apply_lm_head_skip(
    model: torch.nn.Module,
    inner_path: str = "model",
    lm_head_path: str | None = "lm_head",
) -> float:
    """Bypass the LM head of a ``*ForConditionalGeneration``-style encoder.

    ``model.forward`` is replaced with a wrapper that calls the inner base
    model (at ``inner_path``) with the same arguments, so ``hidden_states``
    are produced by exactly the same code path while the full-vocabulary
    logits are never computed. If ``lm_head_path`` names an untied head, it
    is also moved to the CPU.

    Returns the GPU memory freed, in GiB. Idempotent.
    """
    if getattr(model, "_tediet_headless", False):
        return 0.0

    freed = 0
    if lm_head_path is not None:
        lm_head = _resolve(model, lm_head_path)
        tied = any(
            p.data_ptr() in {q.data_ptr() for q in lm_head.parameters()}
            for name, p in model.named_parameters()
            if "embed" in name and not name.startswith(lm_head_path)
        )
        if not tied and next(lm_head.parameters()).device.type != "cpu":
            freed = sum(p.numel() * p.element_size() for p in lm_head.parameters())
            lm_head.to("cpu")

    inner = _resolve(model, inner_path)

    def headless_forward(input_ids=None, attention_mask=None, **kwargs):
        kwargs.pop("labels", None)
        kwargs.setdefault("use_cache", False)
        kwargs.setdefault("return_dict", True)
        return inner(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    model.forward = headless_forward
    model._tediet_headless = True
    return freed / 1024**3


def apply_diet(
    model: torch.nn.Module,
    embed_path: str,
    inner_path: str = "model",
    lm_head_path: str | None = "lm_head",
) -> float:
    """Apply both transformations. Returns total GPU GiB freed."""
    return apply_embed_bridge(model, embed_path) + apply_lm_head_skip(
        model, inner_path=inner_path, lm_head_path=lm_head_path
    )
