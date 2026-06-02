#!/usr/bin/env python3
"""
extract_weights.py — Universal weight extractor for Flash-MoE Universal

Converts HuggingFace safetensors model → binary format for flash-moe inference.
Supports any model size from 1M to 397B+ parameters.

Usage:
    python extract_weights.py --model /path/to/hf/model --output .
    python extract_weights.py --model /path/to/hf/model --output . --config config.json
    python extract_weights.py --model /path/to/hf/model --output . --bits 4
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    print("PyTorch not found. Install with: pip install torch")
    sys.exit(1)

try:
    from safetensors.torch import load_file, safe_open
    HAS_SAFETENSORS = True
except ImportError:
    print("safetensors not found. Install with: pip install safetensors")
    HAS_SAFETENSORS = False

# ─── Quantization ─────────────────────────────────────────────────────────────

def quantize_4bit(tensor: torch.Tensor, group_size: int = 64):
    """
    Quantize a float32 tensor to 4-bit with per-group scale+bias.
    Returns (packed_bytes, scales_fp16, biases_fp16).
    """
    tensor = tensor.float()
    orig_shape = tensor.shape
    # Flatten to 2D: (out_features, in_features)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    rows, cols = tensor.shape
    assert cols % group_size == 0, f"cols ({cols}) not divisible by group_size ({group_size})"

    groups = cols // group_size
    t = tensor.reshape(rows * groups, group_size)

    t_min = t.min(dim=1, keepdim=True).values
    t_max = t.max(dim=1, keepdim=True).values
    scale = (t_max - t_min) / 15.0
    scale = scale.clamp(min=1e-8)
    bias = t_min

    # Quantize to [0, 15]
    t_q = ((t - bias) / scale).round().clamp(0, 15).to(torch.uint8)

    # Pack 2 nibbles per byte
    t_q = t_q.reshape(rows, cols)
    packed = torch.zeros(rows, cols // 2, dtype=torch.uint8)
    packed[:] = t_q[:, 0::2] | (t_q[:, 1::2] << 4)

    scales = scale.reshape(rows, groups).to(torch.float16)
    biases = bias.reshape(rows, groups).to(torch.float16)

    return packed.numpy().tobytes(), scales.numpy().tobytes(), biases.numpy().tobytes()


def quantize_2bit(tensor: torch.Tensor, group_size: int = 64):
    """
    Quantize a float32 tensor to 2-bit with per-group scale+bias (float32).
    Returns (packed_bytes, scales_f32, biases_f32).
    """
    tensor = tensor.float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    rows, cols = tensor.shape
    assert cols % group_size == 0

    groups = cols // group_size
    t = tensor.reshape(rows * groups, group_size)

    t_min = t.min(dim=1, keepdim=True).values
    t_max = t.max(dim=1, keepdim=True).values
    scale = (t_max - t_min) / 3.0
    scale = scale.clamp(min=1e-8)
    bias = t_min

    t_q = ((t - bias) / scale).round().clamp(0, 3).to(torch.uint8)
    t_q = t_q.reshape(rows, cols)

    # Pack 4 x 2-bit per byte
    packed = torch.zeros(rows, cols // 4, dtype=torch.uint8)
    packed[:] = (t_q[:, 0::4]
                | (t_q[:, 1::4] << 2)
                | (t_q[:, 2::4] << 4)
                | (t_q[:, 3::4] << 6))

    scales = scale.reshape(rows, groups).to(torch.float32)
    biases = bias.reshape(rows, groups).to(torch.float32)

    return packed.numpy().tobytes(), scales.numpy().tobytes(), biases.numpy().tobytes()


# ─── Model loading ─────────────────────────────────────────────────────────────

def load_model_index(model_dir: Path):
    """Load model.safetensors.index.json or return None if single file."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            return json.load(f)
    # Single file?
    if (model_dir / "model.safetensors").exists():
        return {"weight_map": {}}
    return None


def get_tensor(model_dir: Path, name: str, weight_map: dict):
    """Load a single tensor from safetensors."""
    if weight_map:
        fname = weight_map.get(name)
        if not fname:
            return None
        fpath = model_dir / fname
    else:
        fpath = model_dir / "model.safetensors"

    if not fpath.exists():
        return None

    with safe_open(str(fpath), framework="pt") as f:
        if name in f.keys():
            return f.get_tensor(name).float()
    return None


# ─── Main extraction ──────────────────────────────────────────────────────────

def extract_weights(model_dir: str, output_dir: str, bits: int = 4, group_size: int = 64):
    model_dir  = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting weights from: {model_dir}")
    print(f"Output to: {output_dir}")
    print(f"Quantization: {bits}-bit, group_size={group_size}")

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print("ERROR: config.json not found")
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    num_layers     = config.get("num_hidden_layers", 32)
    hidden_size    = config.get("hidden_size", 4096)
    num_experts    = config.get("num_experts", 0)
    is_moe         = num_experts > 1

    print(f"Model: {config.get('model_type', 'unknown')}, {num_layers} layers, hidden={hidden_size}")
    print(f"MoE: {'Yes' if is_moe else 'No'} ({num_experts} experts)" if is_moe else "Dense model")

    # Load weight index
    index = load_model_index(model_dir)
    if index is None:
        print("ERROR: No safetensors files found")
        sys.exit(1)

    weight_map = index.get("weight_map", {})

    # Collect all tensor names (skip expert weights — those go in repack_experts.py)
    expert_patterns = [
        "experts.",
        ".mlp.experts.",
        "shared_experts.",
    ]

    def is_expert_weight(name: str) -> bool:
        for p in expert_patterns:
            if p in name: return True
        return False

    # Gather all non-expert tensors
    all_names = list(weight_map.keys()) if weight_map else []
    if not all_names:
        # Single file: enumerate all keys
        sf_path = model_dir / "model.safetensors"
        if sf_path.exists():
            with safe_open(str(sf_path), framework="pt") as f:
                all_names = list(f.keys())

    non_expert_names = [n for n in all_names if not is_expert_weight(n)]
    expert_names     = [n for n in all_names if is_expert_weight(n)]

    print(f"Non-expert tensors: {len(non_expert_names)}")
    print(f"Expert tensors    : {len(expert_names)} (handled by repack_experts.py)")

    # Write non-expert weights to binary file
    out_bin_path  = output_dir / "model_weights.bin"
    out_json_path = output_dir / "model_weights.json"

    manifest = []
    offset = 0
    quant_fn = quantize_4bit if bits == 4 else quantize_2bit

    with open(out_bin_path, "wb") as out_f:
        for i, name in enumerate(non_expert_names):
            print(f"\r  [{i+1}/{len(non_expert_names)}] {name[:60]:<60}", end="")
            tensor = get_tensor(model_dir, name, weight_map)
            if tensor is None:
                print(f"\n  [SKIP] {name} not found")
                continue

            shape = list(tensor.shape)

            # Choose storage format
            if tensor.dtype in (torch.float32, torch.float16) and tensor.ndim >= 2:
                # Quantize to N-bit
                w_bytes, s_bytes, b_bytes = quant_fn(tensor, group_size)

                # Write interleaved: weights, scales, biases
                for part, data in [("weight", w_bytes), ("scales", s_bytes), ("biases", b_bytes)]:
                    # Align to 64 bytes
                    pad = (64 - (offset % 64)) % 64
                    if pad: out_f.write(b'\x00' * pad); offset += pad
                    manifest.append({
                        "name": name.replace(".weight", f".{part}"),
                        "offset": offset, "nbytes": len(data),
                        "shape": shape, "dtype": f"int{bits}_{part}"
                    })
                    out_f.write(data)
                    offset += len(data)
            else:
                # Store as float32 (norms, biases, small tensors)
                data = tensor.numpy().tobytes()
                pad = (64 - (offset % 64)) % 64
                if pad: out_f.write(b'\x00' * pad); offset += pad
                manifest.append({
                    "name": name, "offset": offset,
                    "nbytes": len(data), "shape": shape, "dtype": "float32"
                })
                out_f.write(data)
                offset += len(data)

    print(f"\nWrote {out_bin_path} ({offset/1e9:.2f} GB)")

    # Write manifest
    with open(out_json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {out_json_path} ({len(manifest)} entries)")

    # Copy config.json and tokenizer
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = model_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / fname)
            print(f"Copied: {fname}")

    print("\n✅ Weight extraction complete!")
    print(f"   Non-expert weights: {out_bin_path}")
    if is_moe:
        print(f"   Expert weights: run repack_experts.py next")
    print(f"   Then run: export_tokenizer.py to create tokenizer.bin")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract and quantize model weights for Flash-MoE Universal")
    parser.add_argument("--model",       required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--output",      default=".",   help="Output directory (default: current dir)")
    parser.add_argument("--bits",        type=int, default=4, choices=[2, 4, 8], help="Quantization bits")
    parser.add_argument("--group-size",  type=int, default=64, help="Quantization group size")
    parser.add_argument("--config",      default=None,  help="Override config.json path")
    args = parser.parse_args()

    extract_weights(args.model, args.output, args.bits, args.group_size)


if __name__ == "__main__":
    main()
