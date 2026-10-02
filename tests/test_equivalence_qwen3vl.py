"""Integration test: bit-exactness of diet + stream on a real Qwen3-VL.

Requires a local Qwen-Image 2.x checkout and a CUDA GPU, so this does not run
in CI; it documents the exact validation procedure. Point MODEL_DIR at a
directory holding ``text_encoder/`` and ``processor/``.

Run: TEDIET_MODEL_DIR=/path/to/Qwen-Image-2.1 python tests/test_equivalence_qwen3vl.py
"""
import gc
import os
import sys

import torch

MODEL_DIR = os.environ.get("TEDIET_MODEL_DIR")
DEVICE = os.environ.get("TEDIET_DEVICE", "cuda:0")


def main() -> int:
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    from tediet import apply_diet, apply_stream

    dev = torch.device(DEVICE)
    torch.cuda.set_device(dev)
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        f"{MODEL_DIR}/text_encoder", dtype=torch.bfloat16, local_files_only=True
    ).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/processor", local_files_only=True)
    ids = tok(
        "A serene Japanese garden with a red bridge", return_tensors="pt"
    ).input_ids.to(dev)

    with torch.no_grad():
        ref = te(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[-1].clone()
    resident = torch.cuda.memory_allocated(dev) / 2**30

    freed = apply_diet(te, embed_path="model.language_model.embed_tokens")
    pinned = apply_stream(te, layers_path="model.language_model.layers", device=dev, window=2)
    gc.collect()
    torch.cuda.empty_cache()
    now = torch.cuda.memory_allocated(dev) / 2**30
    print(f"resident {resident:.2f} -> {now:.2f} GiB (diet freed {freed:.2f}, pinned {pinned:.2f})")

    with torch.no_grad():
        out1 = te(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[-1]
        out2 = te(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[-1]
    ok1, ok2 = torch.equal(ref, out1), torch.equal(ref, out2)
    print(f"bit-identical: first={ok1} second={ok2}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    if MODEL_DIR is None:
        print("set TEDIET_MODEL_DIR to a Qwen-Image 2.x directory")
        sys.exit(2)
    sys.exit(main())
