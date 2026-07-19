# ResearchMind — System Architecture

Local multimodal RAG pipeline: ingests text, documents, audio, video and
spreadsheets, embeds them into ChromaDB, and synthesizes a structured
research summary through a local Ollama LLM — no external API calls
for the core Summarize flow. (The Similarity Finder is the one
exception — see `similarity-finder-architecture.md`.)

## Full pipeline

```mermaid
flowchart TB
    subgraph CLIENT["CLIENT"]
        UI["research_summary_app.html<br/>upload UI + research-context field"]
    end

    subgraph API["FASTAPI · api.py"]
        EP1["POST /api/summarize"]
        EP2["GET /api/health"]
        EP3["GET /api/models"]
    end

    subgraph ORCH["ORCHESTRATOR · main.py"]
        VALIDATE["1 · validate file paths"]
        LLMCHECK["2 · LLMClient.is_available()"]
        VSINIT["3 · VectorStore.reset()"]
        ROUTE["4 · route_files()"]
        PARALLEL["5 · ThreadPoolExecutor\nparallel_workers = 2"]
    end

    subgraph ROUTER["utils/file_router.py"]
        RMAP["EXTENSION_MAP\ntxt·md·html → text\npdf·docx·pptx·odt → document\nmp3·wav·flac → audio\nmp4·mkv·mov → video\ncsv·xlsx·ods → spreadsheet"]
    end

    subgraph PROCESSORS["PROCESSORS/  (5 modality handlers)"]
        TXT["text_processor.py"]
        DOC["document_processor.py"]
        AUD["audio_processor.py"]
        VID["video_processor.py"]
        SHEET["spreadsheet_processor.py"]
    end

    subgraph LOCALMODELS["LOCAL MODELS"]
        WHISPER["faster-whisper\nCTranslate2 · CUDA/CPU auto"]
        FFMPEG["FFmpeg\n16kHz mono WAV extraction"]
        PANDAS["pandas / numpy\ndescribe() + corr()"]
    end

    subgraph STORE["core/vector_store.py"]
        CHUNK["_chunk_text()\n1000 chars / 150 overlap"]
        EMBED["sentence-transformers\nall-MiniLM-L6-v2"]
        CHROMA[("ChromaDB\nPersistentClient · cosine HNSW")]
    end

    subgraph SYNTH["core/summarizer.py"]
        COMBINE["build_combined_content()\nbudget ≈12,000 chars total"]
        RAG["query() top-6 chunks\nby user_context"]
        PARSE["_parse_summary()\nsection-keyword parser"]
    end

    subgraph LLM["core/llm_client.py"]
        CLIENT["LLMClient\nurllib → REST, no SDK"]
    end

    subgraph OLLAMA["OLLAMA · localhost:11434"]
        MODEL["mistral / llama3 / phi3\nGPU-accelerated inference"]
    end

    subgraph CFG["config.py + utils/gpu_check.py"]
        GPUDET["_detect_gpu() via nvidia-smi\nauto-selects Whisper size + compute type"]
    end

    UI -->|"multipart: files[] + context"| EP1
    EP1 --> VALIDATE --> LLMCHECK --> VSINIT --> ROUTE
    ROUTE --> RMAP --> PARALLEL
    PARALLEL --> TXT & DOC & AUD & VID & SHEET

    AUD --> WHISPER
    VID --> FFMPEG --> WHISPER
    SHEET --> PANDAS

    TXT --> CHUNK
    DOC --> CHUNK
    AUD --> CHUNK
    VID --> CHUNK
    SHEET --> CHUNK
    CHUNK --> EMBED --> CHROMA

    TXT & DOC & AUD & VID & SHEET --> COMBINE
    CHROMA -.->|"semantic retrieval\nonly if user_context set"| RAG
    RAG --> COMBINE
    COMBINE --> CLIENT
    CLIENT -->|"POST /api/generate"| MODEL
    MODEL --> CLIENT --> PARSE
    PARSE --> EP1
    EP1 -->|"JSON: topic, overview,\nkeyPoints, keywords …"| UI

    GPUDET -.configures.-> WHISPER
    GPUDET -.configures.-> EMBED
    GPUDET -.OLLAMA_MODEL env.-> MODEL
```

## Request sequence

```mermaid
sequenceDiagram
    participant U as Browser
    participant A as api.py
    participant P as run_pipeline()
    participant R as file_router
    participant Proc as Processors (parallel)
    participant V as VectorStore
    participant S as summarizer.py
    participant O as Ollama

    U->>A: POST /api/summarize (files[], context)
    A->>A: save uploads to temp dir
    A->>P: run_pipeline(paths, context)
    P->>O: is_available()
    O-->>P: model status
    P->>V: reset() collection
    P->>R: route_files(paths)
    R-->>P: grouped by category
    par parallel processing, 2 workers
        P->>Proc: process(text / document files)
        P->>Proc: process(audio / video / sheet files)
        Proc->>O: generate() per-file summary / gist
        O-->>Proc: LLM text
    end
    Proc-->>P: list[result dicts]
    P->>V: add_document() per source
    P->>S: build_combined_content(results, context)
    opt user_context provided
        S->>V: query(context, n_results=6)
        V-->>S: top-k relevant chunks
    end
    S->>O: generate_final_summary(combined)
    O-->>S: structured summary text
    S-->>P: parsed {topic, overview, keyPoints, ...}
    P-->>A: result dict + processingLog
    A->>A: cleanup temp dir
    A-->>U: JSON response
```

## Compute placement

| Component | GPU | Mechanism | Fallback |
|---|---|---|---|
| Ollama LLM | yes | Detects NVIDIA automatically | CPU inference (slower) |
| faster-whisper (ASR) | yes | CTranslate2 CUDA backend | CPU, int8 compute type |
| sentence-transformers | yes, verified | Real encode() probe before committing to CUDA | CPU if the probe fails |
| ChromaDB | no | Vector ops fast enough on CPU | — |
| pandas / numpy | no | CPU-only statistical analysis | optional cuDF for scale |
