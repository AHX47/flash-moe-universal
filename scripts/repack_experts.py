#!/usr/bin/env python3
"""
repack_experts.py — Universal MoE expert repacker for Flash-MoE Universal

Converts HuggingFace MoE expert weights → per-layer packed binary files.
Supports: Qwen3.5-397B, Mixtral 8x7B, DeepSeek-V2, and any MoE model.

Usage:
    python repack_experts.py --model /path/to/model --output ./packed_experts
    python repack_experts.py --model /path/to/model --output . --bits 2
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
    from safetensors.torch import safe_open
except ImportError:
    print("Install: pip install torch safetensors")
    sys.exit(1)


# ─── Quantization (same as extract_weights.py) ───────────────────────────────

def quantize_4bit_expert(w_gate, w_up, w_down, group_size=64):
    """
    Pack one expert's gate/up/down projections into 4-bit binary blob.
    Layout: [gate_w|gate_scales|gate_biases|up_w|up_scales|up_biases|down_w|down_scales|down_biases]
    """
    def quant(t):
        t = t.float()
        rows, cols = t.shape
        assert cols % group_size == 0
        groups = cols // group_size
        tr = t.reshape(rows * groups, group_size)
        t_min = tr.min(dim=1, keepdim=True).values
        t_max = tr.max(dim=1, keepdim=True).values
        scale = (t_max - t_min) / 15.0
        scale = scale.clamp(min=1e-8)
        bias  = t_min
        t_q   = ((tr - bias) / scale).round().clamp(0, 15).to(torch.uint8)
        t_q   = t_q.reshape(rows, cols)
        packed = torch.zeros(rows, cols // 2, dtype=torch.uint8)
        packed[:] = t_q[:, 0::2] | (t_q[:, 1::2] << 4)
        sc = scale.reshape(rows, groups).to(torch.float16)
        bi = bias.reshape(rows, groups).to(torch.float16)
        return packed.numpy().tobytes(), sc.numpy().tobytes(), bi.numpy().tobytes()

    blob = b""
    for w in [w_gate, w_up, w_down]:
        wbytes, sbytes, bbytes = quant(w)
        blob += wbytes + sbytes + bbytes
    return blob


def quantize_2bit_expert(w_gate, w_up, w_down, group_size=64):
    """Pack one expert to 2-bit."""
    def quant(t):
        t = t.float()
        rows, cols = t.shape
        assert cols % group_size == 0
        groups = cols // group_size
        tr = t.reshape(rows * groups, group_size)
        t_min = tr.min(dim=1, keepdim=True).values
        t_max = tr.max(dim=1, keepdim=True).values
        scale = (t_max - t_min) / 3.0
        scale = scale.clamp(min=1e-8)
        bias  = t_min
        t_q   = ((tr - bias) / scale).round().clamp(0, 3).to(torch.uint8)
        t_q   = t_q.reshape(rows, cols)
        packed = torch.zeros(rows, cols // 4, dtype=torch.uint8)
        packed[:] = (t_q[:, 0::4] | (t_q[:, 1::4] << 2) | (t_q[:, 2::4] << 4) | (t_q[:, 3::4] << 6))
        sc = scale.reshape(rows, groups).to(torch.float32)
        bi = bias.reshape(rows, groups).to(torch.float32)
        return packed.numpy().tobytes(), sc.numpy().tobytes(), bi.numpy().tobytes()

    blob = b""
    for w in [w_gate, w_up, w_down]:
        wbytes, sbytes, bbytes = quant(w)
        blob += wbytes + sbytes + bbytes
    return blob


# ─── Expert name patterns for common architectures ────────────────────────────

EXPERT_PATTERNS = {
    # Qwen3.5 MoE
    "qwen3_moe": {
        "gate": "model.layers.{L}.mlp.experts.{E}.gate_proj.weight",
        "up":   "model.layers.{L}.mlp.experts.{E}.up_proj.weight",
        "down": "model.layers.{L}.mlp.experts.{E}.down_proj.weight",
    },
    # Mixtral
    "mixtral": {
        "gate": "model.layers.{L}.block_sparse_moe.experts.{E}.w1.weight",
        "up":   "model.layers.{L}.block_sparse_moe.experts.{E}.w3.weight",
        "down": "model.layers.{L}.block_sparse_moe.experts.{E}.w2.weight",
    },
    # DeepSeek-V2
    "deepseek_v2": {
        "gate": "model.layers.{L}.mlp.experts.{E}.gate_proj.weight",
        "up":   "model.layers.{L}.mlp.experts.{E}.up_proj.weight",
        "down": "model.layers.{L}.mlp.experts.{E}.down_proj.weight",
    },
    # Generic / fallback
    "generic": {
        "gate": "model.layers.{L}.mlp.experts.{E}.gate_proj.weight",
        "up":   "model.layers.{L}.mlp.experts.{E}.up_proj.weight",
        "down": "model.layers.{L}.mlp.experts.{E}.down_proj.weight",
    },
}


def detect_arch(config: dict) -> str:
    model_type = config.get("model_type", "").lower()
    if "qwen" in model_type:
        return "qwen3_moe"
    if "mixtral" in model_type or "mistral" in model_type:
        return "mixtral"
    if "deepseek" in model_type:
        return "deepseek_v2"
    return "generic"


def get_tensor(model_dir: Path, name: str, weight_map: dict, file_cache: dict):
    """Load a tensor, caching open file handles."""
    if weight_map:
        fname = weight_map.get(name)
        if not fname:
            return None
        fpath = model_dir / fname
    else:
        fpath = model_dir / "model.safetensors"

    if str(fpath) not in file_cache:
        if not fpath.exists():
            return None
        file_cache[str(fpath)] = safe_open(str(fpath), framework="pt")

    f = file_cache[str(fpath)]
    try:
        return f.get_tensor(name).float()
    except Exception:
        return None


# ─── Main repacking ───────────────────────────────────────────────────────────

def repack_experts(model_dir: str, output_dir: str, bits: int = 4, group_size: int = 64):
    model_dir  = Path(model_dir)
    output_dir = Path(output_dir)
    experts_dir = output_dir / f"packed_experts{'_2bit' if bits == 2 else ''}"
    experts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repacking experts from: {model_dir}")
    print(f"Output: {experts_dir}")
    print(f"Quantization: {bits}-bit, group_size={group_size}")

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    num_layers  = config.get("num_hidden_layers", 32)
    num_experts = config.get("num_experts", config.get("num_local_experts", 8))
    arch        = detect_arch(config)
    patterns    = EXPERT_PATTERNS.get(arch, EXPERT_PATTERNS["generic"])

    print(f"Architecture : {arch}")
    print(f"Layers       : {num_layers}")
    print(f"Experts/layer: {num_experts}")

    # Load weight index
    index_file = model_dir / "model.safetensors.index.json"
    weight_map = {}
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f).get("weight_map", {})

    quant_fn = quantize_4bit_expert if bits == 4 else quantize_2bit_expert
    file_cache = {}
    expert_size = None

    total_bytes = 0
    for layer in range(num_layers):
        out_path = experts_dir / f"layer_{layer:02d}.bin"
        if out_path.exists():
            existing = out_path.stat().st_size
            if existing > 0:
                print(f"  Layer {layer:02d}: already exists ({existing/1e9:.2f} GB), skipping")
                continue

        print(f"\r  Layer {layer:02d}/{num_layers-1}: packing {num_experts} experts...", end="", flush=True)

        layer_blob = b""
        for expert in range(num_experts):
            gate_name = patterns["gate"].format(L=layer, E=expert)
            up_name   = patterns["up"].format(L=layer, E=expert)
            down_name = patterns["down"].format(L=layer, E=expert)

            w_gate = get_tensor(model_dir, gate_name, weight_map, file_cache)
            w_up   = get_tensor(model_dir, up_name,   weight_map, file_cache)
            w_down = get_tensor(model_dir, down_name, weight_map, file_cache)

            if w_gate is None or w_up is None or w_down is None:
                # Try transposed names (some models store transposed)
                for suffix in [".weight", ""]:
                    w_gate = w_gate or get_tensor(model_dir, gate_name + suffix, weight_map, file_cache)
                if w_gate is None:
                    print(f"\n  [WARN] Expert {expert} of layer {layer} not found, padding with zeros")
                    # Pad with zeros of expected size
                    if expert_size:
                        layer_blob += b'\x00' * expert_size
                    continue

            blob = quant_fn(w_gate, w_up, w_down, group_size)

            if expert_size is None:
                expert_size = len(blob)
                print(f"\n  Expert size: {expert_size:,} bytes ({expert_size/1e6:.2f} MB)")
                print(f"  Layer size : {expert_size * num_experts / 1e9:.3f} GB")

            layer_blob += blob

        with open(out_path, "wb") as f:
            f.write(layer_blob)
        total_bytes += len(layer_blob)
        print(f"\r  Layer {layer:02d}: wrote {len(layer_blob)/1e9:.3f} GB  ", end="")

    # Close all file handles
    for fh in file_cache.values():
        try: fh.__exit__(None, None, None)
        except: pass

    print(f"\n\n✅ Expert repacking complete!")
    print(f"   Total: {total_bytes/1e9:.1f} GB in {experts_dir}")
    print(f"   Expert size: {expert_size:,} bytes" if expert_size else "")

    # Save metadata
    meta = {
        "arch": arch,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "bits": bits,
        "group_size": group_size,
        "expert_size_bytes": expert_size,
    }
    with open(experts_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   Metadata: {experts_dir / 'meta.json'}")


def main():
    parser = argparse.ArgumentParser(description="Repack MoE expert weights for Flash-MoE Universal")
    parser.add_argument("--model",      required=True, help="HuggingFace model directory")
    parser.add_argument("--output",     default=".",   help="Output directory")
    parser.add_argument("--bits",       type=int, default=4, choices=[2, 4], help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument("--layer",      type=int, default=None, help="Pack only this layer (for testing)")
    args = parser.parse_args()
    repack_experts(args.model, args.output, args.bits, args.group_size)

if __name__ == "__main__":
    main()
