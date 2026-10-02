"""Reproduce tediet's text-encoder claims on one machine with one command.

The script loads the text encoder fresh for each configuration (the
transformations are irreversible in-process), then reports for each:

- GPU residency after setup
- encode time (CUDA events, mean over --runs after one warmup)
- peak GPU memory during the encode
- whether the hidden states are bit-identical to the fully resident reference

Configurations:

- resident            : untouched baseline (also produces the reference output)
- tediet              : apply_diet + apply_stream
- tediet-diet-only    : apply_diet alone
- group-offload-stream: diffusers apply_group_offloading(use_stream=True)
- group-offload       : diffusers apply_group_offloading(use_stream=False)

Example (Qwen-Image 2.x checkout):

    python benchmarks/bench_text_encoder.py \
        --recipe qwen3vl --model-dir /path/to/Qwen-Image-2.1 --device cuda:0

Results are printed as a Markdown table and written to
benchmarks/results_<recipe>.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RECIPES = {
    "qwen3vl": {
        "subdir": "text_encoder",
        "tokenizer_subdir": "processor",
        "loader": "Qwen3VLForConditionalGeneration",
        "embed_path": "model.language_model.embed_tokens",
        "layers_path": "model.language_model.layers",
        "inner_path": "model",
        "lm_head_path": "lm_head",
        "offload_root_path": "model.language_model",
    },
    "gemma": {
        "subdir": "text_encoder_bnb_4bit",
        "tokenizer_subdir": "tokenizer",
        "loader": "Gemma4UnifiedForConditionalGeneration",
        "embed_path": "model.language_model.embed_tokens",
        "layers_path": "model.language_model.layers",
        "inner_path": "model",
        "lm_head_path": "lm_head",
        "offload_root_path": "model.language_model",
        # leaf_level does not hook bitsandbytes Linear4bit modules and crashes
        "group_offload_type": "block_level",
    },
    "t5": {
        "subdir": "",
        "tokenizer_subdir": "",
        "loader": "T5EncoderModel",
        "embed_path": "encoder.embed_tokens",
        "layers_path": "encoder.block",
        "inner_path": None,
        "lm_head_path": None,
        "offload_root_path": "encoder",
    },
}

PROMPT = (
    "A serene Japanese garden with a red wooden bridge over a koi pond, "
    "autumn maple leaves, soft morning light, photorealistic"
)


def load_encoder(recipe: dict, model_dir: Path, device: torch.device):
    import transformers

    cls = getattr(transformers, recipe["loader"])
    # device_map covers bitsandbytes checkpoints too, where .to(device) is unsupported
    te = cls.from_pretrained(
        model_dir / recipe["subdir"], dtype=torch.bfloat16, local_files_only=True,
        device_map={"": str(device)},
    ).eval()
    tok = transformers.AutoTokenizer.from_pretrained(
        model_dir / recipe["tokenizer_subdir"], local_files_only=True
    )
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    return te, ids


def encode(te, ids):
    with torch.no_grad():
        out = te(input_ids=ids, output_hidden_states=True, return_dict=True)
    return out.hidden_states[-1]


def measure(te, ids, device, runs: int):
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for i in range(runs + 1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = encode(te, ids)
        end.record()
        torch.cuda.synchronize(device)
        if i > 0:
            times.append(start.elapsed_time(end) / 1000)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    return sum(times) / len(times), peak, out


def setup_config(name: str, te, device, recipe: dict, window: int):
    if name == "resident":
        return
    if name in ("tediet", "tediet-diet-only"):
        from tediet import apply_diet, apply_embed_bridge, apply_stream

        if recipe["inner_path"] is None:
            apply_embed_bridge(te, recipe["embed_path"])
        else:
            apply_diet(
                te,
                embed_path=recipe["embed_path"],
                inner_path=recipe["inner_path"],
                lm_head_path=recipe["lm_head_path"],
            )
        if name == "tediet":
            apply_stream(te, recipe["layers_path"], device=device, window=window)
        return
    if name.startswith("group-offload"):
        from diffusers.hooks import apply_group_offloading

        # The fairest stock configuration we found: leaf_level on the inner
        # language model. block_level applied to the whole encoder treats the
        # entire base model as one group (peak = full model), and block_level
        # applied to inner submodules crashes on the unhooked embedding.
        root = te
        for part in recipe["offload_root_path"].split("."):
            root = getattr(root, part)
        kwargs = {"offload_type": recipe.get("group_offload_type", "leaf_level")}
        if kwargs["offload_type"] == "block_level":
            kwargs["num_blocks_per_group"] = 1
        apply_group_offloading(
            root,
            onload_device=device,
            offload_device=torch.device("cpu"),
            use_stream=name.endswith("stream"),
            **kwargs,
        )
        return
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["resident", "tediet", "tediet-diet-only", "group-offload-stream", "group-offload"],
    )
    parser.add_argument(
        "--out-suffix",
        default="",
    )
    args = parser.parse_args()

    recipe = RECIPES[args.recipe]
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    reference = None
    rows = []
    for name in args.configs:
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        te, ids = load_encoder(recipe, args.model_dir, device)
        load_s = time.perf_counter() - t0
        try:
            setup_config(name, te, device, recipe, args.window)
            gc.collect()
            torch.cuda.empty_cache()
            resident = torch.cuda.memory_allocated(device) / 2**30
            mean_s, peak, out = measure(te, ids, device, args.runs)
            if name == "resident":
                reference = out.clone()
            exact = bool(torch.equal(reference, out)) if reference is not None else None
            rows.append(
                {
                    "config": name,
                    "resident_gib": round(resident, 2),
                    "encode_s": round(mean_s, 4),
                    "peak_gib": round(peak, 2),
                    "bit_identical": exact,
                    "load_s": round(load_s, 1),
                }
            )
        except Exception as exc:  # report, keep going
            rows.append({"config": name, "error": f"{type(exc).__name__}: {exc}"})
        del te
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n| config | resident GiB | encode s | peak GiB | bit-identical |")
    print("|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['config']} | failed: {r['error']} | | | |")
        else:
            print(
                f"| {r['config']} | {r['resident_gib']} | {r['encode_s']} "
                f"| {r['peak_gib']} | {r['bit_identical']} |"
            )

    out_path = Path(__file__).parent / f"results_{args.recipe}{args.out_suffix}.json"
    meta = {
        "recipe": args.recipe,
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "window": args.window,
        "runs": args.runs,
        "rows": rows,
    }
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
