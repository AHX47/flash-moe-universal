#!/usr/bin/env python3
"""
download_model.py — Download + prepare any model for Flash-MoE Universal

Downloads from HuggingFace Hub and automatically:
  1. Downloads the model
  2. Extracts weights to 4-bit binary
  3. Repacks MoE experts (if MoE model)
  4. Exports tokenizer

Usage:
    python scripts/download_model.py                          # interactive menu
    python scripts/download_model.py --model qwen2.5-0.5b    # by preset name
    python scripts/download_model.py --hf Qwen/Qwen2.5-0.5B-Instruct
    python scripts/download_model.py --hf Qwen/Qwen2.5-0.5B-Instruct --outdir ./my-model
    python scripts/download_model.py --list                   # show all presets
"""

import os
import sys
import json
import subprocess
import argparse
import shutil
from pathlib import Path

# ─── Model catalog ───────────────────────────────────────────────────────────
# Format: (preset_name, hf_repo_id, size_gb, ram_gb, is_moe, description)
MODELS = [
    # ── Tiny / Test models (great for testing) ──────────────────────────────
    ("qwen2.5-0.5b",  "Qwen/Qwen2.5-0.5B-Instruct",                0.5,  1,  False, "Smallest Qwen — ideal for testing"),
    ("qwen2.5-1.5b",  "Qwen/Qwen2.5-1.5B-Instruct",                1.0,  2,  False, "Good quality, runs on 2GB RAM"),
    ("qwen2.5-3b",    "Qwen/Qwen2.5-3B-Instruct",                  2.0,  4,  False, "Great quality/speed balance"),
    ("llama3.2-1b",   "meta-llama/Llama-3.2-1B-Instruct",          1.0,  2,  False, "Tiny Llama — fast CPU test"),
    ("llama3.2-3b",   "meta-llama/Llama-3.2-3B-Instruct",          2.0,  4,  False, "Small Llama — good for CPU"),
    ("smollm2-360m",  "HuggingFaceTB/SmolLM2-360M-Instruct",       0.4,  1,  False, "Ultra tiny — 360M params, <500MB"),
    ("smollm2-1.7b",  "HuggingFaceTB/SmolLM2-1.7B-Instruct",       1.0,  2,  False, "Small but capable"),
    ("phi3-mini",     "microsoft/Phi-3-mini-4k-instruct",           2.0,  4,  False, "Microsoft Phi-3 mini, 3.8B"),
    ("gemma2-2b",     "google/gemma-2-2b-it",                       2.0,  3,  False, "Google Gemma 2 2B"),
    # ── Medium models ────────────────────────────────────────────────────────
    ("qwen2.5-7b",    "Qwen/Qwen2.5-7B-Instruct",                  4.5,  8,  False, "Strong 7B model"),
    ("mistral-7b",    "mistralai/Mistral-7B-Instruct-v0.3",         4.5,  8,  False, "Mistral 7B"),
    ("llama3.1-8b",   "meta-llama/Meta-Llama-3.1-8B-Instruct",     4.5,  8,  False, "Llama 3.1 8B"),
    ("phi4",          "microsoft/phi-4",                            10.0, 12, False, "Phi-4 14B — excellent reasoning"),
    ("qwen2.5-14b",   "Qwen/Qwen2.5-14B-Instruct",                  9.0, 12, False, "Qwen 14B"),
    # ── Large models ─────────────────────────────────────────────────────────
    ("qwen2.5-32b",   "Qwen/Qwen2.5-32B-Instruct",                 19.0, 24, False, "Qwen 32B — needs 24GB+"),
    ("qwen2.5-72b",   "Qwen/Qwen2.5-72B-Instruct",                 40.0, 48, False, "Qwen 72B — needs 48GB+"),
    ("llama3.1-70b",  "meta-llama/Meta-Llama-3.1-70B-Instruct",    40.0, 48, False, "Llama 70B — needs 48GB+"),
    # ── MoE models ───────────────────────────────────────────────────────────
    ("mixtral-8x7b",  "mistralai/Mixtral-8x7B-Instruct-v0.1",      26.0, 16, True,  "Mixtral MoE — 12.9B active"),
    ("qwen3.5-397b",  "mlx-community/Qwen3.5-397B-A17B-4bit",     209.0, 48, True,  "⚡ ORIGINAL flash-moe model — 48GB RAM"),
]


def get_ram_gb():
    """Get available RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 8.0  # assume 8GB if unknown


def check_disk_space(path: Path, needed_gb: float) -> bool:
    """Check if enough disk space."""
    try:
        stat = os.statvfs(str(path))
        free_gb = stat.f_bavail * stat.f_frsize / 1e9
        return free_gb >= needed_gb
    except Exception:
        return True  # assume OK


def print_catalog(filter_ram=None):
    """Print available models."""
    ram = get_ram_gb()
    print(f"\n{'#':<4} {'Name':<20} {'Size':>8} {'RAM':>6} {'Type':<8} Description")
    print("─" * 80)
    for i, (name, hf, size, needed_ram, is_moe, desc) in enumerate(MODELS):
        ok    = "✅" if needed_ram <= ram else "⚠️ "
        mtype = "MoE" if is_moe else "Dense"
        if filter_ram and needed_ram > filter_ram:
            continue
        print(f"[{i:<2}] {name:<20} {size:>6.1f}GB {needed_ram:>4}GB {mtype:<8} {desc}  {ok}")
    print(f"\n  Your RAM: ~{ram:.0f} GB")
    print("  ✅ = fits in RAM,  ⚠️  = may be tight")


def find_model(query: str):
    """Find model by preset name (fuzzy)."""
    q = query.lower().replace("-", "").replace(".", "")
    for m in MODELS:
        n = m[0].lower().replace("-", "").replace(".", "")
        if q in n or n in q:
            return m
    return None


def check_hf_cli():
    """Check if huggingface-cli is available, install if not."""
    if shutil.which("huggingface-cli"):
        return True
    print("[INFO] huggingface-cli not found, trying pip install...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "huggingface_hub[cli]", "-q"],
        capture_output=True
    )
    return result.returncode == 0 and shutil.which("huggingface-cli") is not None


def download_hf(hf_id: str, output_dir: Path, token: str = None) -> bool:
    """Download model from HuggingFace Hub."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try huggingface-cli first
    if shutil.which("huggingface-cli"):
        cmd = ["huggingface-cli", "download", hf_id,
               "--local-dir", str(output_dir),
               "--local-dir-use-symlinks", "False"]
        if token:
            cmd += ["--token", token]
        print(f"[DOWNLOAD] {hf_id}")
        print(f"[TARGET]   {output_dir}")
        print("[CMD]     ", " ".join(cmd))
        print()
        result = subprocess.run(cmd)
        return result.returncode == 0

    # Try Python API
    try:
        from huggingface_hub import snapshot_download
        print(f"[DOWNLOAD] {hf_id} → {output_dir}")
        snapshot_download(
            repo_id=hf_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        return True
    except ImportError:
        print("[ERROR] Neither huggingface-cli nor huggingface_hub Python package found.")
        print("        Install with: pip install huggingface_hub[cli]")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return False


def prepare_model(model_dir: Path, is_moe: bool, bits: int = 4):
    """Run extract_weights and repack_experts on downloaded model."""
    scripts_dir = Path(__file__).parent

    print(f"\n{'='*60}")
    print(f"[PREPARE] Extracting weights ({bits}-bit)...")
    print(f"{'='*60}")

    # extract_weights.py
    extract_script = scripts_dir / "extract_weights.py"
    if extract_script.exists():
        cmd = [sys.executable, str(extract_script),
               "--model", str(model_dir),
               "--output", str(model_dir),
               "--bits", str(bits)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("[WARN] extract_weights.py failed — model may not be in safetensors format")
    else:
        print(f"[WARN] {extract_script} not found, skipping weight extraction")

    # repack_experts.py (MoE only)
    if is_moe:
        print(f"\n{'='*60}")
        print(f"[PREPARE] Repacking MoE experts ({bits}-bit)...")
        print(f"{'='*60}")
        repack_script = scripts_dir / "repack_experts.py"
        if repack_script.exists():
            cmd = [sys.executable, str(repack_script),
                   "--model", str(model_dir),
                   "--output", str(model_dir),
                   "--bits", str(bits)]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("[WARN] repack_experts.py failed")

    # export_tokenizer.py
    tokenizer_json = model_dir / "tokenizer.json"
    tokenizer_bin  = model_dir / "tokenizer.bin"
    if tokenizer_json.exists() and not tokenizer_bin.exists():
        export_script = scripts_dir / "export_tokenizer.py"
        if export_script.exists():
            print(f"\n[PREPARE] Exporting tokenizer...")
            subprocess.run([sys.executable, str(export_script),
                            str(tokenizer_json), str(tokenizer_bin)])

    print(f"\n✅ Model ready at: {model_dir}")


def interactive_menu():
    """Show interactive download menu."""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║      Flash-MoE Universal — Model Downloader            ║")
    print("╚" + "═"*58 + "╝")

    print_catalog()

    print("\nOptions:")
    print("  Enter a number [0-{}] to select a model".format(len(MODELS)-1))
    print("  Or type a preset name (e.g. 'qwen2.5-0.5b')")
    print("  Or type a HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct')")
    print("  Press Ctrl+C to cancel")

    try:
        choice = input("\n> Your choice: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(0)

    # Numeric index
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(MODELS):
            return MODELS[idx]
        else:
            print(f"[ERROR] Invalid index: {idx}")
            sys.exit(1)

    # Preset name
    m = find_model(choice)
    if m:
        return m

    # Raw HF repo ID
    if "/" in choice:
        return (choice.split("/")[-1].lower(), choice, 0, 0, False, "Custom model")

    print(f"[ERROR] Not found: '{choice}'")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare models for Flash-MoE Universal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_model.py                         # Interactive menu
  python scripts/download_model.py --model smollm2-360m   # Tiny 360M test model
  python scripts/download_model.py --model qwen2.5-0.5b   # Smallest Qwen
  python scripts/download_model.py --model qwen2.5-1.5b   # Good balance
  python scripts/download_model.py --model llama3.2-1b    # Tiny Llama
  python scripts/download_model.py --hf HuggingFaceTB/SmolLM2-360M-Instruct
  python scripts/download_model.py --list                  # Show all models
        """
    )
    parser.add_argument("--model",   help="Preset model name (e.g. qwen2.5-0.5b)")
    parser.add_argument("--hf",      help="HuggingFace repo ID")
    parser.add_argument("--outdir",  default="./models", help="Output directory (default: ./models)")
    parser.add_argument("--bits",    type=int, default=4, choices=[2, 4], help="Quantization bits (default: 4)")
    parser.add_argument("--no-prep", action="store_true", help="Skip weight preparation step")
    parser.add_argument("--token",   help="HuggingFace access token (for gated models)")
    parser.add_argument("--list",    action="store_true", help="List all available models")
    args = parser.parse_args()

    if args.list:
        print_catalog()
        return

    # Determine which model to download
    if args.hf:
        name = args.hf.split("/")[-1].lower()
        m = next((x for x in MODELS if x[1] == args.hf), None)
        if not m:
            m = (name, args.hf, 0, 0, False, "Custom model")
    elif args.model:
        m = find_model(args.model)
        if not m:
            print(f"[ERROR] Unknown model preset: '{args.model}'")
            print("        Run with --list to see available presets")
            print("        Or use --hf <repo_id> for any HuggingFace model")
            sys.exit(1)
    else:
        m = interactive_menu()

    name, hf_id, size_gb, ram_gb, is_moe, desc = m

    # Print summary
    ram = get_ram_gb()
    print(f"\n{'='*60}")
    print(f"  Model  : {name}")
    print(f"  HF ID  : {hf_id}")
    print(f"  Size   : ~{size_gb:.1f} GB")
    print(f"  RAM    : ~{ram_gb} GB needed  (you have ~{ram:.0f} GB)")
    print(f"  Type   : {'MoE (Mixture of Experts)' if is_moe else 'Dense'}")
    print(f"  Info   : {desc}")
    print(f"{'='*60}\n")

    if ram_gb > ram * 1.1:
        print(f"⚠️  Warning: This model needs ~{ram_gb}GB RAM but you only have ~{ram:.0f}GB")
        print("    It may run slowly using swap space.")
        try:
            ans = input("   Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                print("Cancelled.")
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    # Output directory
    out_base = Path(args.outdir)
    model_dir = out_base / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check disk space
    if size_gb > 0 and not check_disk_space(out_base, size_gb * 1.2):
        print(f"⚠️  Warning: Need ~{size_gb*1.2:.0f}GB free disk space")

    # Check/install huggingface-cli
    if not check_hf_cli():
        print("\n[ERROR] Cannot install huggingface-cli automatically.")
        print("  Manual install: pip install huggingface_hub[cli]")
        print(f"  Then run: huggingface-cli download {hf_id} --local-dir {model_dir}")
        sys.exit(1)

    # Download
    print(f"[DOWNLOADING] {hf_id}")
    ok = download_hf(hf_id, model_dir, token=args.token)

    if not ok:
        print("[ERROR] Download failed.")
        print(f"  Try manually: huggingface-cli download {hf_id} --local-dir {model_dir}")
        sys.exit(1)

    print(f"\n✅ Downloaded to: {model_dir}")

    # Prepare weights
    if not args.no_prep:
        prepare_model(model_dir, is_moe, bits=args.bits)
    else:
        print("[SKIP] Weight preparation skipped (--no-prep)")

    # Print run instructions
    binary = Path(__file__).parent.parent / "build" / "flash_moe"
    print(f"\n{'='*60}")
    print("🚀 Ready to run!")
    print(f"{'='*60}")
    print(f"\n  {binary} --model {model_dir} --info")
    print(f"  {binary} --model {model_dir} --prompt 'Hello, how are you?'")
    print(f"  {binary} --model {model_dir} --chat")
    print(f"  {binary} --model {model_dir} --serve 8080")
    print(f"\n  # Or launch the GUI:")
    print(f"  python gui/app.py")


if __name__ == "__main__":
    main()
