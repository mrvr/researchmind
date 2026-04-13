"""
utils/gpu_check.py
──────────────────────────────────────────────────────────────────
Standalone GPU diagnostics script for ResearchMind.

Run before starting the pipeline to verify:
  - NVIDIA driver + CUDA availability
  - VRAM capacity and recommended model sizes
  - PyTorch CUDA availability (for sentence-transformers)
  - faster-whisper CUDA availability
  - Ollama GPU offloading status

Usage:
    python utils/gpu_check.py
"""

import subprocess
import sys


# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK   = f"{GREEN}✓{RESET}"
WARN = f"{YELLOW}⚠{RESET}"
FAIL = f"{RED}✗{RESET}"

def _h(title: str):
    print(f"\n{BOLD}{CYAN}── {title} {'─'*(50-len(title))}{RESET}")

def _ok(msg):   print(f"  {OK}  {msg}")
def _warn(msg): print(f"  {WARN}  {YELLOW}{msg}{RESET}")
def _fail(msg): print(f"  {FAIL}  {RED}{msg}{RESET}")
def _info(msg): print(f"     {msg}")


# ── 1. nvidia-smi ─────────────────────────────────────────────────────────────
def check_nvidia_smi():
    _h("NVIDIA Driver & GPU")
    try:
        raw = subprocess.check_output(["nvidia-smi"], text=True, timeout=5)

        # Parse driver + CUDA version from header
        for line in raw.splitlines():
            if "Driver Version" in line:
                parts = line.split("|")
                for p in parts:
                    p = p.strip()
                    if "Driver Version" in p or "CUDA Version" in p:
                        _ok(p)

        # Per-GPU details
        gpu_out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        gpus = []
        for line in gpu_out.strip().splitlines():
            idx, name, total, free, temp, util = [x.strip() for x in line.split(",")]
            total_mb = int(total)
            free_mb  = int(free)
            gpus.append({"idx": idx, "name": name, "total_mb": total_mb, "free_mb": free_mb})
            _ok(f"GPU {idx}: {name}")
            _info(f"VRAM total : {total_mb:,} MB  ({total_mb/1024:.1f} GB)")
            _info(f"VRAM free  : {free_mb:,} MB  ({free_mb/1024:.1f} GB)")
            _info(f"Temp / Util: {temp}°C / {util}%")

        return gpus

    except FileNotFoundError:
        _fail("nvidia-smi not found — NVIDIA driver not installed")
        _info("Install from: https://ubuntu.com/server/docs/nvidia-drivers-installation")
        return []
    except Exception as e:
        _fail(f"nvidia-smi error: {e}")
        return []


# ── 2. PyTorch CUDA ───────────────────────────────────────────────────────────
def check_pytorch():
    _h("PyTorch (sentence-transformers backend)")
    try:
        import torch
        _ok(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            _ok(f"CUDA available: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                _ok(f"  Device {i}: {props.name}  |  {props.total_memory // 1024**2:,} MB VRAM")
        else:
            _warn("PyTorch CUDA not available — sentence-transformers will use CPU")
            _info("Fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        _warn("PyTorch not installed — will be installed with requirements.txt")


# ── 3. faster-whisper ─────────────────────────────────────────────────────────
def check_faster_whisper():
    _h("faster-whisper")
    try:
        from faster_whisper import WhisperModel
        _ok("faster-whisper installed")

        # Try loading the tiny model on CUDA just to verify
        try:
            import ctranslate2
            _ok(f"CTranslate2 version: {ctranslate2.__version__}")
            cuda_support = ctranslate2.get_cuda_device_count() > 0
            if cuda_support:
                _ok(f"CTranslate2 CUDA devices: {ctranslate2.get_cuda_device_count()}")
            else:
                _warn("CTranslate2 reports 0 CUDA devices — Whisper will use CPU")
                _info("Fix: pip install ctranslate2 --extra-index-url https://download.pytorch.org/whl/cu121")
        except Exception as e:
            _warn(f"Could not check CTranslate2 CUDA: {e}")

    except ImportError:
        _warn("faster-whisper not installed — will be installed with requirements.txt")


# ── 4. Ollama GPU status ───────────────────────────────────────────────────────
def check_ollama():
    _h("Ollama LLM")
    try:
        import urllib.request, json
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        _ok(f"Ollama is running — {len(models)} model(s) available")
        for m in models:
            _info(f"  • {m}")
        if not models:
            _warn("No models pulled yet — run: ollama pull mistral")
    except Exception:
        _fail("Ollama not running or not installed")
        _info("Install: curl -fsSL https://ollama.ai/install.sh | sh")
        _info("Start  : ollama serve")
        _info("Pull   : ollama pull mistral")


# ── 5. VRAM-based recommendations ─────────────────────────────────────────────
def print_recommendations(gpus: list):
    _h("Recommended Configuration for Your Hardware")
    if not gpus:
        _warn("No GPU detected — CPU-only mode")
        _info("Whisper model : base (fastest on CPU)")
        _info("Ollama model  : phi3:mini or tinyllama")
        _info("Embeddings    : all-MiniLM-L6-v2 (CPU)")
        return

    vram_mb = gpus[0]["total_mb"]
    name    = gpus[0]["name"]
    print(f"  Based on: {BOLD}{name}{RESET} ({vram_mb:,} MB / {vram_mb/1024:.1f} GB VRAM)\n")

    # Whisper
    if vram_mb >= 10_000:
        whisper = "large-v3  (best accuracy)"
    elif vram_mb >= 5_000:
        whisper = "medium    (great balance)"
    elif vram_mb >= 2_000:
        whisper = "small     (fast + decent)"
    else:
        whisper = "base      (minimal footprint)"

    # Ollama
    if vram_mb >= 24_000:
        ollama = "mistral:7b-instruct  or  llama3:8b"
    elif vram_mb >= 12_000:
        ollama = "mistral:7b-instruct  or  phi3:medium"
    elif vram_mb >= 6_000:
        ollama = "phi3:mini  or  gemma2:2b"
    else:
        ollama = "tinyllama  or  phi3:mini"

    _ok(f"Whisper model  →  {whisper}")
    _ok(f"Ollama model   →  {ollama}")
    _ok(f"Embeddings     →  all-MiniLM-L6-v2 (GPU-accelerated via PyTorch)")
    _ok(f"Compute type   →  float16 (CUDA)")

    print(f"\n  {BOLD}Set in config.py or via environment:{RESET}")
    print(f"    export WHISPER_MODEL=large-v3")
    print(f"    export OLLAMA_MODEL=mistral")
    print(f"    ollama pull mistral && ollama serve")


# ── 6. FFmpeg ─────────────────────────────────────────────────────────────────
def check_ffmpeg():
    _h("FFmpeg (video audio extraction)")
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5)
        version_line = out.splitlines()[0]
        _ok(version_line)
    except FileNotFoundError:
        _fail("FFmpeg not found")
        _info("Install: sudo apt install ffmpeg")
    except Exception as e:
        _fail(f"FFmpeg error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}{'='*55}")
    print("  ResearchMind — GPU & Dependency Diagnostics")
    print(f"{'='*55}{RESET}")

    gpus = check_nvidia_smi()
    check_pytorch()
    check_faster_whisper()
    check_ffmpeg()
    check_ollama()
    print_recommendations(gpus)

    print(f"\n{BOLD}{'='*55}{RESET}\n")
