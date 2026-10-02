"""Windowed layer streaming with a fixed ring buffer.

The decoder layers of the text encoder are kept in pinned host memory and
copied to the GPU only while an encode is running, a few layers ahead of the
compute stream. The GPU side uses ``window + 1`` fixed buffer slots that are
reused in a ring, so an encode allocates **zero** new GPU memory — which keeps
the allocator stable even next to CUDA Graph memory pools, where on-demand
allocation would fragment and pile up reservations.

Synchronization contract:

- The copy stream waits on ``slot_free[slot]`` (recorded by the compute stream
  when the layer that last used the slot finished) before overwriting a slot.
- The compute stream waits on ``copied[i]`` (recorded by the copy stream)
  before running layer ``i``.
- ``slots = window + 1`` guarantees the slot being copied into is never the
  slot the current layer is computing from.

With a prefetch window of 2 the per-layer host-to-device copy hides completely
behind the previous layer's compute on a PCIe 4.0 x16 link, and the output is
bit-identical to the fully resident model (the weights are untouched; only
their location changes).

Supported parameter types: plain tensors (fp32/bf16/fp16) and bitsandbytes
4-bit layers (the ``quant_state`` tensors travel with the layer).

Constraint: the model must otherwise stay resident (no accelerate offload
hooks), and encodes must not run concurrently from multiple threads.
"""
from __future__ import annotations

import torch


def _collect_movable(module: torch.nn.Module):
    """Enumerate device-movable tensors of ``module`` as (owner, attr, tensor).

    Covers ``nn.Parameter.data``, named buffers, and bitsandbytes
    ``Params4bit.quant_state`` tensors (absmax / code / offset, plus the
    nested ``state2`` when double quantization is on).
    """
    items = []
    for _, p in module.named_parameters(recurse=True):
        items.append((p, "data", p.data))
        qs = getattr(p, "quant_state", None)
        if qs is not None:
            for attr in ("absmax", "code", "offset"):
                t = getattr(qs, attr, None)
                if isinstance(t, torch.Tensor):
                    items.append((qs, attr, t))
            nested = getattr(qs, "state2", None)
            if nested is not None:
                for attr in ("absmax", "code"):
                    t = getattr(nested, attr, None)
                    if isinstance(t, torch.Tensor):
                        items.append((nested, attr, t))
    for name, b in module.named_buffers(recurse=True):
        owner = module
        parts = name.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        items.append((owner, parts[-1], b))
    return items


class WindowedLayerStreamer:
    def __init__(self, layers, device: torch.device, window: int = 2):
        self.layers = list(layers)
        self.device = torch.device(device)
        self.window = max(1, int(window))
        self.nslots = self.window + 1
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copied = [torch.cuda.Event() for _ in self.layers]
        self.cpu_snap = []
        pinned_bytes = 0
        for layer in self.layers:
            snap = []
            for owner, attr, t in _collect_movable(layer):
                cpu = t.detach().to("cpu")
                if not cpu.is_pinned():
                    cpu = cpu.pin_memory()
                pinned_bytes += cpu.numel() * cpu.element_size()
                snap.append((owner, attr, cpu))
                self._set(owner, attr, cpu)
            self.cpu_snap.append(snap)
        self.pinned_gib = pinned_bytes / 1024**3

        # Layers are grouped by structural signature; each signature gets its
        # own ring of window+1 slots. Models with alternating layer kinds
        # (for example Gemma's sliding/global attention layers) therefore
        # work too: a layer only ever reuses a slot of its own signature.
        self.sig_of = []
        occurrence = []
        counts: dict[tuple, int] = {}
        for snap in self.cpu_snap:
            sig = tuple((tuple(c.shape), c.dtype) for _, _, c in snap)
            self.sig_of.append(sig)
            occurrence.append(counts.get(sig, 0))
            counts[sig] = counts.get(sig, 0) + 1
        self.slot_of = [occ % self.nslots for occ in occurrence]
        self.slots: dict[tuple, list[list[torch.Tensor]]] = {}
        self.slot_free: dict[tuple, list[torch.cuda.Event]] = {}
        slots_bytes = 0
        for i, snap in enumerate(self.cpu_snap):
            sig = self.sig_of[i]
            if sig not in self.slots:
                rings = []
                for _ in range(min(self.nslots, counts[sig])):
                    bufs = [torch.empty(c.shape, dtype=c.dtype, device=self.device) for _, _, c in snap]
                    slots_bytes += sum(t.numel() * t.element_size() for t in bufs)
                    rings.append(bufs)
                self.slots[sig] = rings
                evs = [torch.cuda.Event() for _ in rings]
                for ev in evs:
                    ev.record()
                self.slot_free[sig] = evs
        self.slots_gib = slots_bytes / 1024**3

        for i, layer in enumerate(self.layers):
            layer.register_forward_pre_hook(self._pre(i))
            layer.register_forward_hook(self._post(i))

    @staticmethod
    def _set(owner, attr, tensor):
        if isinstance(owner, torch.nn.Parameter):
            owner.data = tensor
        else:
            setattr(owner, attr, tensor)

    def _onload(self, i: int) -> None:
        sig, slot_idx = self.sig_of[i], self.slot_of[i]
        slot_idx %= len(self.slots[sig])
        slot = self.slots[sig][slot_idx]
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(self.slot_free[sig][slot_idx])
            for (owner, attr, cpu), buf in zip(self.cpu_snap[i], slot):
                buf.copy_(cpu, non_blocking=True)
                self._set(owner, attr, buf)
            self.copied[i].record(self.copy_stream)

    def _pre(self, i: int):
        def hook(_m, _args):
            torch.cuda.current_stream(self.device).wait_event(self.copied[i])
        return hook

    def _post(self, i: int):
        def hook(_m, _args, _out):
            for owner, attr, cpu in self.cpu_snap[i]:
                self._set(owner, attr, cpu)
            sig, slot_idx = self.sig_of[i], self.slot_of[i]
            self.slot_free[sig][slot_idx % len(self.slots[sig])].record(
                torch.cuda.current_stream(self.device)
            )
            nxt = i + self.window
            if nxt < len(self.layers):
                self._onload(nxt)
        return hook

    def begin(self) -> None:
        for i in range(min(self.window, len(self.layers))):
            self._onload(i)

    def end(self) -> None:
        # Weights never change on the GPU, so restoring is pointer reassignment:
        # no device-to-host copy happens here.
        torch.cuda.current_stream(self.device).synchronize()
        for snap in self.cpu_snap:
            for owner, attr, cpu in snap:
                self._set(owner, attr, cpu)


def apply_stream(
    model: torch.nn.Module,
    layers_path: str,
    device: torch.device | str,
    window: int = 2,
) -> float:
    """Stream the layer stack at ``layers_path`` and wrap ``model.forward``
    in ``begin()``/``end()``. Composes with :func:`tediet.apply_diet` applied
    before or after. Returns the pinned host memory used, in GiB. Idempotent.
    """
    if getattr(model, "_tediet_streamed", False):
        return 0.0

    layers = model
    for part in layers_path.split("."):
        layers = getattr(layers, part)
    streamer = WindowedLayerStreamer(layers, device=device, window=window)
    inner_forward = model.forward

    def streamed_forward(*args, **kwargs):
        streamer.begin()
        try:
            return inner_forward(*args, **kwargs)
        finally:
            streamer.end()

    model.forward = streamed_forward
    model._tediet_streamed = True
    model._tediet_streamer = streamer
    return streamer.pinned_gib
