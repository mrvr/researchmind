# ResearchMind — Local AI Pipeline

A fully local, modular Python pipeline that processes PDFs, documents, audio, video, 
spreadsheets and generates a unified research summary using a local LLM via Ollama + ChromaDB.
**Optimised for NVIDIA GPU (CUDA) on Ubuntu.**

---

## Project Structure

```
researchmind/
├── README.md
├── requirements.txt
├── config.py                     ← Central config (auto-detects GPU, picks best Whisper model)
├── main.py                       ← Pipeline orchestrator (CLI + importable library)
├── api.py                        ← FastAPI server (connects frontend → pipeline)
│
├── processors/
│   ├── text_processor.py         ← TXT, MD, RTF, HTML
│   ├── document_processor.py     ← PDF, DOCX, ODT, PPTX
│   ├── audio_processor.py        ← MP3, WAV, M4A, FLAC — faster-whisper (CUDA)
│   ├── video_processor.py        ← MP4, MKV, AVI, MOV — FFmpeg → Whisper (CUDA)
│   └── spreadsheet_processor.py  ← XLS, XLSX, CSV — pandas analysis → LLM
│
├── core/
│   ├── vector_store.py           ← ChromaDB + sentence-transformers (GPU embeddings)
│   ├── llm_client.py             ← Ollama HTTP client (GPU inference via Ollama)
│   └── summarizer.py             ← Merge all outputs → final LLM summary
│
└── utils/
    ├── file_router.py            ← Extension → processor mapping
    ├── gpu_check.py              ← GPU diagnostics + model recommendations ← RUN THIS FIRST
    └── logger.py                 ← Coloured console logger
```

---

## Prerequisites

### 1. NVIDIA Driver + CUDA Toolkit
```bash
# Check your driver is installed
nvidia-smi

# Install CUDA toolkit if needed (Ubuntu 22.04)
sudo apt install nvidia-cuda-toolkit
```

### 2. Install Ollama (uses GPU automatically)
```bash
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model — choose based on your VRAM (gpu_check.py recommends the best one):
ollama pull mistral          # 7B — good for ≥8GB VRAM
# ollama pull llama3         # 8B — slightly better quality
# ollama pull phi3:mini      # 3.8B — good for 4-6GB VRAM
# ollama pull gemma2:2b      # 2B — minimal VRAM

ollama serve                 # starts on http://localhost:11434
```

### 3. Install FFmpeg
```bash
sudo apt update && sudo apt install -y ffmpeg
```

### 4. Python Environment (install order matters for CUDA!)
```bash
python3 -m venv venv
source venv/bin/activate

# Step A: PyTorch with CUDA 12.1 FIRST (needed by sentence-transformers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step B: All remaining dependencies
pip install -r requirements.txt
```

### 5. Verify everything works
```bash
python utils/gpu_check.py
```
This prints your GPU name, VRAM, CUDA version, and recommends the best Whisper + Ollama model.

---

## Running

### As a FastAPI server (for the frontend webapp):
```bash
source venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Then update the frontend's `callBackendAPI()` to point to `http://localhost:8000/api/summarize`.

### As a CLI tool:
```bash
python main.py --files paper.pdf lecture.mp4 data.xlsx --context "Focus on methodology"
```

### Health check (verify all deps + Ollama):
```
GET http://localhost:8000/api/health
```

---

## Configuration (config.py)

| Setting | Default | Notes |
|---|---|---|
| `OLLAMA_MODEL` | `mistral` | Override: `export OLLAMA_MODEL=llama3` |
| `WHISPER_MODEL_SIZE` | **auto** | Chosen by VRAM: tiny→base→small→medium→large-v3 |
| `WHISPER_DEVICE` | **auto** | `cuda` if GPU detected, else `cpu` |
| `WHISPER_COMPUTE_TYPE` | **auto** | `float16` on CUDA, `int8` on CPU |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Runs on GPU via PyTorch |
| `CHUNK_SIZE` | `1000` chars | ChromaDB chunk size |
| `CHUNK_OVERLAP` | `150` chars | Overlap between chunks |

All settings can also be overridden via environment variables.

---

## GPU Behaviour Summary

| Component | GPU Used? | How |
|---|---|---|
| Ollama LLM | ✅ Yes | Ollama detects NVIDIA automatically |
| faster-whisper | ✅ Yes | CTranslate2 CUDA backend |
| sentence-transformers | ✅ Yes | PyTorch CUDA |
| ChromaDB | ❌ No | CPU only (vector ops are fast enough) |
| pandas / data analysis | ❌ No | CPU (optional: add cuDF for large datasets) |
