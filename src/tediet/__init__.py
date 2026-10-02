"""tediet: put your diffusion pipeline's text encoder on a diet.

Two composable, output-exact VRAM reducers for resident text encoders:

- :func:`apply_diet` moves the embedding table and the (unused) LM head to
  the CPU and skips the logits computation.
- :func:`apply_stream` keeps the decoder layers in pinned host memory and
  streams them to fixed GPU ring-buffer slots only while an encode runs.
"""
from .diet import apply_diet, apply_embed_bridge, apply_lm_head_skip
from .stream import WindowedLayerStreamer, apply_stream

__version__ = "0.1.0"

__all__ = [
    "apply_diet",
    "apply_embed_bridge",
    "apply_lm_head_skip",
    "apply_stream",
    "WindowedLayerStreamer",
]
