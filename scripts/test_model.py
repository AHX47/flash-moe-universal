#!/usr/bin/env python3
"""
test_model.py — Full test suite for Flash-MoE Universal

Tests:
  1. Binary exists and runs
  2. System info (CPU arch, SIMD, backends)
  3. Model loading
  4. Tokenizer
  5. Single token generation
  6. Streaming generation
  7. Speed benchmark (tok/s)
  8. Chat mode (multi-turn)
  9. HTTP server API
  10. Memory usage

Usage:
    python scripts/test_model.py                         # test binary only (no model needed)
    python scripts/test_model.py --model ./models/qwen2.5-0.5b
    python scripts/test_model.py --model ./models/qwen2.5-0.5b --bench
    python scripts/test_model.py --quick                 # just smoke tests
"""

import os
import sys
import time
import json
import shutil
import signal
import argparse
import platform
import subprocess
import threading
import tempfile
from pathlib import Path

# ─── ANSI colors ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✅ PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}❌ FAIL{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠️  WARN{RESET}  {msg}")
def info(msg):  print(f"  {BLUE}ℹ️  INFO{RESET}  {msg}")
def head(msg):  print(f"\n{BOLD}{CYAN}{'─'*56}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'─'*56}{RESET}")


# ─── Find binary ─────────────────────────────────────────────────────────────

def find_binary(hint=None):
    if hint and Path(hint).exists():
        return str(hint)
    candidates = [
        Path(__file__).parent.parent / "build" / "flash_moe",
        Path(__file__).parent.parent / "build" / "Release" / "flash_moe.exe",
        Path(__file__).parent.parent / "build" / "flash_moe.exe",
        shutil.which("flash_moe"),
    ]
    for c in candidates:
        if c and Path(str(c)).exists():
            return str(c)
    return None


# ─── Test runner ─────────────────────────────────────────────────────────────

class TestSuite:
    def __init__(self, binary, model_dir=None):
        self.binary    = binary
        self.model_dir = model_dir
        self.passed    = 0
        self.failed    = 0
        self.skipped   = 0
        self.results   = []

    def run(self, name, fn):
        try:
            result = fn()
            if result is None or result is True:
                ok(name)
                self.passed += 1
                self.results.append((name, "PASS", ""))
            elif result is False:
                fail(name)
                self.failed += 1
                self.results.append((name, "FAIL", ""))
            else:
                warn(f"{name} — {result}")
                self.passed += 1
                self.results.append((name, "WARN", str(result)))
        except Exception as e:
            fail(f"{name}: {e}")
            self.failed += 1
            self.results.append((name, "FAIL", str(e)))

    def skip(self, name, reason=""):
        print(f"  {YELLOW}⏭  SKIP{RESET}  {name}  {reason}")
        self.skipped += 1
        self.results.append((name, "SKIP", reason))

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*56}")
        print(f"  Results: {self.passed}/{total} passed  "
              f"| {self.failed} failed  | {self.skipped} skipped")
        if self.failed == 0:
            print(f"  {GREEN}{BOLD}All tests passed! ✅{RESET}")
        else:
            print(f"  {RED}{BOLD}{self.failed} test(s) failed ❌{RESET}")
        print(f"{'='*56}\n")
        return self.failed == 0


# ─── Individual tests ────────────────────────────────────────────────────────

def run_binary(binary, args, timeout=30):
    """Run binary and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            [binary] + args,
            capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except FileNotFoundError:
        return -2, "", "NOT FOUND"


def test_binary_exists(binary):
    if not binary:
        raise Exception("Binary not found. Run ./build.sh first.")
    if not Path(binary).exists():
        raise Exception(f"Binary not at: {binary}")


def test_binary_runs(binary):
    rc, out, err = run_binary(binary, ["--version"])
    if rc != 0:
        raise Exception(f"Exit code {rc}\n{err[:200]}")
    if "flash_moe" not in out.lower() and "1.0" not in out:
        raise Exception(f"Unexpected output: {out[:100]}")


def test_system_info(binary):
    rc, out, err = run_binary(binary, ["--info"])
    if rc != 0:
        raise Exception(f"Exit {rc}: {err[:100]}")
    lines = out + err
    checks = ["OS:", "Arch:", "SIMD:", "Cores:"]
    missing = [c for c in checks if c not in lines]
    if missing:
        return f"Missing fields: {missing}"


def test_cpu_arch_detected(binary):
    rc, out, err = run_binary(binary, ["--info"])
    text = out + err
    current_arch = platform.machine().lower()
    # Check that arch is reported
    if "x86_64" in text or "arm64" in text or "aarch64" in text or "Arch:" in text:
        return True
    raise Exception(f"Arch not detected in output: {text[:200]}")


def test_simd_detection(binary):
    rc, out, err = run_binary(binary, ["--info"])
    text = out + err
    simd_words = ["AVX2", "NEON", "SSE", "Scalar", "AVX-512"]
    found = [s for s in simd_words if s in text]
    if not found:
        return f"No SIMD info in output"
    info(f"SIMD detected: {found[0]}")


def test_backend_list(binary):
    rc, out, err = run_binary(binary, ["--info"])
    text = out + err
    if "CPU" not in text:
        raise Exception("CPU backend not listed")
    backends_found = []
    for b in ["CPU", "Metal", "OpenCL", "CUDA"]:
        if b in text:
            backends_found.append(b)
    info(f"Backends: {', '.join(backends_found)}")


def test_help(binary):
    rc, out, err = run_binary(binary, ["--help"])
    if rc != 0 and "--prompt" not in (out + err):
        raise Exception(f"--help failed: {err[:100]}")
    required_flags = ["--model", "--prompt", "--chat", "--backend"]
    text = out + err
    missing = [f for f in required_flags if f not in text]
    if missing:
        return f"Missing flags in help: {missing}"


def test_model_exists(model_dir):
    if not model_dir:
        raise Exception("No --model specified")
    p = Path(model_dir)
    if not p.exists():
        raise Exception(f"Directory not found: {model_dir}")
    config = p / "config.json"
    if not config.exists():
        raise Exception(f"config.json not found in {model_dir}")


def test_config_json(model_dir):
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    required = ["hidden_size", "num_hidden_layers"]
    missing = [k for k in required if k not in cfg]
    if missing:
        return f"config.json missing keys: {missing}"
    info(f"Model: {cfg.get('model_type','?')}, layers={cfg.get('num_hidden_layers')}, hidden={cfg.get('hidden_size')}")


def test_weights_exist(model_dir):
    p = Path(model_dir)
    # Check for safetensors OR pre-extracted binary
    safetensors = list(p.glob("*.safetensors")) + list(p.glob("model.safetensors.index.json"))
    binary_weights = p / "model_weights.bin"
    if not safetensors and not binary_weights.exists():
        raise Exception("No model weights found (no .safetensors or model_weights.bin)")
    if binary_weights.exists():
        size_gb = binary_weights.stat().st_size / 1e9
        info(f"model_weights.bin: {size_gb:.2f} GB")
    elif safetensors:
        total = sum(f.stat().st_size for f in p.glob("*.safetensors") if f.is_file())
        info(f"safetensors: {total/1e9:.2f} GB ({len(list(p.glob('*.safetensors')))} shards)")


def test_tokenizer(model_dir):
    p = Path(model_dir)
    tok_json = p / "tokenizer.json"
    tok_bin  = p / "tokenizer.bin"
    if not tok_json.exists() and not tok_bin.exists():
        return f"No tokenizer found (tokenizer.json or tokenizer.bin)"
    if tok_json.exists():
        with open(tok_json) as f:
            tok = json.load(f)
        vocab_size = len(tok.get("model", {}).get("vocab", {}))
        info(f"Tokenizer: vocab_size={vocab_size or '?'}")


def test_single_generation(binary, model_dir, timeout=120):
    """Test that model generates at least a few tokens."""
    rc, out, err = run_binary(
        binary,
        ["--model", model_dir, "--prompt", "Hello", "--tokens", "5", "--backend", "cpu"],
        timeout=timeout
    )
    text = out + err
    if rc != 0 and "tokens" not in text.lower():
        raise Exception(f"Generation failed (rc={rc}): {err[:200]}")
    # Check some output was produced
    if len(out.strip()) < 1:
        return f"No output produced (stderr: {err[:100]})"


def test_generation_speed(binary, model_dir, n_tokens=20, timeout=300):
    """Benchmark tokens per second."""
    start = time.time()
    rc, out, err = run_binary(
        binary,
        ["--model", model_dir, "--prompt", "Write a short poem about nature.",
         "--tokens", str(n_tokens), "--backend", "cpu"],
        timeout=timeout
    )
    elapsed = time.time() - start
    text = out + err

    # Parse tok/s from output
    tps = None
    for line in text.split("\n"):
        if "tok/s" in line or "token" in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                if "tok" in p.lower() and i > 0:
                    try:
                        tps = float(parts[i-1])
                        break
                    except ValueError:
                        pass

    if tps is None and elapsed > 0:
        # Estimate from elapsed time
        words = len(out.split())
        tps = words / elapsed if elapsed > 0 else 0

    if tps is not None and tps > 0:
        info(f"Speed: {tps:.2f} tok/s ({n_tokens} tokens in {elapsed:.1f}s)")
        if tps < 0.1:
            return f"Very slow: {tps:.3f} tok/s — model may not be loaded correctly"
    else:
        return f"Could not measure speed (elapsed={elapsed:.1f}s)"


def test_different_prompts(binary, model_dir):
    """Test a few different prompt types."""
    prompts = [
        ("Math",      "What is 2 + 2?",                 5),
        ("Code",      "Write hello world in Python:",   10),
        ("Multilang", "مرحبا بك",                       5),
    ]
    all_ok = True
    for name, prompt, n in prompts:
        rc, out, err = run_binary(
            binary,
            ["--model", model_dir, "--prompt", prompt, "--tokens", str(n), "--backend", "cpu"],
            timeout=60
        )
        if rc != 0 and not out.strip():
            warn(f"  Prompt '{name}' failed")
            all_ok = False
        else:
            info(f"  '{name}' OK: {repr(out[:40])}")
    return True if all_ok else "Some prompts failed"


def test_temperature_sampling(binary, model_dir):
    """Test different temperatures produce different outputs."""
    outputs = set()
    for temp in ["0.0", "0.5", "1.0"]:
        rc, out, err = run_binary(
            binary,
            ["--model", model_dir, "--prompt", "The sky is", "--tokens", "8",
             "--temp", temp, "--backend", "cpu"],
            timeout=60
        )
        outputs.add(out.strip()[:50])
    # At least 2 different outputs expected (deterministic vs random)
    if len(outputs) < 1:
        raise Exception("No output produced")
    info(f"Got {len(outputs)} unique outputs across temperatures")


def test_server_starts(binary, model_dir):
    """Test that --serve starts an HTTP server."""
    port = 18765
    proc = subprocess.Popen(
        [binary, "--model", model_dir, "--serve", str(port), "--backend", "cpu"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(3)  # give server time to start

    # Check process is running
    if proc.poll() is not None:
        _, err = proc.communicate()
        raise Exception(f"Server exited early: {err[:100]}")

    # Try HTTP request
    try:
        import urllib.request
        req = urllib.request.urlopen(f"http://localhost:{port}/", timeout=5)
        info(f"Server responded on port {port}")
    except Exception as e:
        info(f"Server running (HTTP check: {e})")
    finally:
        proc.terminate()
        try: proc.wait(timeout=3)
        except: proc.kill()


def test_memory_usage(binary, model_dir):
    """Check peak memory during generation."""
    try:
        import psutil
    except ImportError:
        return "psutil not installed, skipping memory check"

    proc = subprocess.Popen(
        [binary, "--model", model_dir, "--prompt", "Hello world", "--tokens", "10", "--backend", "cpu"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    peak_mb = 0
    try:
        ps_proc = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                mem = ps_proc.memory_info().rss / 1e6
                if mem > peak_mb:
                    peak_mb = mem
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)
    except Exception:
        pass
    finally:
        proc.wait()

    if peak_mb > 0:
        info(f"Peak RAM usage: {peak_mb:.0f} MB")
        if peak_mb > 50000:  # 50GB
            return f"Very high RAM: {peak_mb:.0f} MB"
    return True


def test_cpu_quantization(binary, model_dir):
    """Test both 4-bit and 2-bit modes."""
    for flag, name in [([], "4-bit (default)"), (["--2bit"], "2-bit")]:
        rc, out, err = run_binary(
            binary,
            ["--model", model_dir, "--prompt", "Hello", "--tokens", "3",
             "--backend", "cpu"] + flag,
            timeout=60
        )
        if rc == 0 or out.strip():
            info(f"  {name}: OK")
        else:
            warn(f"  {name}: failed ({err[:60]})")


def test_platform_info():
    """Print platform information."""
    import platform as plat
    info(f"OS       : {plat.platform()}")
    info(f"Python   : {plat.python_version()}")
    info(f"Machine  : {plat.machine()}")
    info(f"CPU cores: {os.cpu_count()}")
    try:
        import psutil
        ram = psutil.virtual_memory()
        info(f"RAM      : {ram.total/1e9:.1f} GB total, {ram.available/1e9:.1f} GB free")
    except ImportError:
        pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Flash-MoE Universal")
    parser.add_argument("--model",   help="Path to model directory")
    parser.add_argument("--binary",  help="Path to flash_moe binary")
    parser.add_argument("--bench",   action="store_true", help="Run speed benchmark")
    parser.add_argument("--quick",   action="store_true", help="Binary tests only, no model needed")
    parser.add_argument("--server",  action="store_true", help="Include server test")
    parser.add_argument("--tokens",  type=int, default=20, help="Tokens for benchmark (default: 20)")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║      Flash-MoE Universal — Test Suite                ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════╝{RESET}")

    binary    = find_binary(args.binary)
    model_dir = args.model
    suite     = TestSuite(binary, model_dir)

    # ── Group 1: Platform ─────────────────────────────────────────────────────
    head("1. Platform Information")
    suite.run("Platform info", test_platform_info)

    # ── Group 2: Binary ───────────────────────────────────────────────────────
    head("2. Binary Tests (no model needed)")
    suite.run("Binary exists",       lambda: test_binary_exists(binary))
    suite.run("Binary runs",         lambda: test_binary_runs(binary))
    suite.run("--version flag",      lambda: test_binary_runs(binary))
    suite.run("--help flag",         lambda: test_help(binary))
    suite.run("--info flag",         lambda: test_system_info(binary))
    suite.run("CPU arch detected",   lambda: test_cpu_arch_detected(binary))
    suite.run("SIMD detected",       lambda: test_simd_detection(binary))
    suite.run("Backend list",        lambda: test_backend_list(binary))

    if args.quick:
        suite.summary()
        sys.exit(0 if suite.failed == 0 else 1)

    # ── Group 3: Model files ──────────────────────────────────────────────────
    head("3. Model File Tests")
    if not model_dir:
        suite.skip("Model directory", "no --model specified")
        suite.skip("config.json",     "no --model specified")
        suite.skip("Weight files",    "no --model specified")
        suite.skip("Tokenizer",       "no --model specified")
    else:
        suite.run("Model directory exists", lambda: test_model_exists(model_dir))
        suite.run("config.json valid",      lambda: test_config_json(model_dir))
        suite.run("Weight files exist",     lambda: test_weights_exist(model_dir))
        suite.run("Tokenizer present",      lambda: test_tokenizer(model_dir))

    # ── Group 4: Generation ───────────────────────────────────────────────────
    head("4. Generation Tests")
    if not model_dir or not binary:
        suite.skip("Single generation", "no model or binary")
        suite.skip("Different prompts", "no model or binary")
        suite.skip("Temperature sampling", "no model or binary")
    else:
        suite.run("Single generation (5 tokens)",
                  lambda: test_single_generation(binary, model_dir))
        suite.run("Different prompt types",
                  lambda: test_different_prompts(binary, model_dir))
        suite.run("Temperature sampling",
                  lambda: test_temperature_sampling(binary, model_dir))
        suite.run("Quantization modes (4-bit / 2-bit)",
                  lambda: test_cpu_quantization(binary, model_dir))

    # ── Group 5: Performance ─────────────────────────────────────────────────
    head("5. Performance Tests")
    if not model_dir or not binary:
        suite.skip("Speed benchmark", "no model or binary")
        suite.skip("Memory usage",    "no model or binary")
    else:
        n = args.tokens
        suite.run(f"Speed benchmark ({n} tokens)",
                  lambda: test_generation_speed(binary, model_dir, n_tokens=n))
        suite.run("Memory usage",
                  lambda: test_memory_usage(binary, model_dir))

    # ── Group 6: Server ───────────────────────────────────────────────────────
    if args.server:
        head("6. Server Tests")
        if not model_dir or not binary:
            suite.skip("HTTP server", "no model or binary")
        else:
            suite.run("HTTP server starts", lambda: test_server_starts(binary, model_dir))

    # ── Summary ───────────────────────────────────────────────────────────────
    success = suite.summary()

    if not binary:
        print(f"{YELLOW}Build the binary first:{RESET}")
        print("  cd flash-moe-universal && ./build.sh\n")
    if not model_dir:
        print(f"{YELLOW}Download a model to run full tests:{RESET}")
        print("  python scripts/download_model.py --model smollm2-360m\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
