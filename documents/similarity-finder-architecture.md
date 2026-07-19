# Similarity Finder — Architecture

Plagiarism/originality check for uploaded research papers. Extends
ResearchMind (currently a fully-local pipeline) with a new stage that
searches external academic sources and flags **≥60% content similarity**
as a **chance of plagiarism**.

> **Local vs. external:** every other stage of ResearchMind runs fully
> offline. This feature is the one exception — it must reach external
> search APIs over the network. It is designed as an isolated, opt-in
> pipeline stage so the rest of the app stays local-only.

## Source coverage decision

Google Scholar and the ACM Digital Library have no official public
search API. Rather than scrape them (fragile, against ToS), this design
uses **official/free APIs only**, per the hybrid approach:

| Requested source | Connector used | Why |
|---|---|---|
| Google Scholar | *(not implemented)* | No public API; SerpAPI would be the only reliable path, out of scope for the official-API-only approach. Known coverage gap. |
| IEEE site | `ieee_connector.py` — IEEE Xplore API | Official API, free developer key, metadata + abstracts. |
| ACM site | *(not implemented directly)* | No public ACM DL API. Covered indirectly — Crossref and Semantic Scholar both index ACM-published papers (title/abstract/DOI). |
| AAS site | `ads_connector.py` — NASA ADS API | AAS journals (ApJ, AJ, ApJL) are canonically indexed by NASA's Astrophysics Data System, which has a free, official API — the correct on-topic source for "AAS site". |
| general web / broad coverage | `arxiv_connector.py`, `semantic_scholar_connector.py`, `crossref_connector.py` | All free, keyless or low-friction, and together give broad cross-publisher coverage (including most IEEE/ACM/AAS metadata as a fallback). |

## Component architecture

```mermaid
flowchart TB
    subgraph INPUT["INPUT"]
        PAPER["uploaded research paper\n(PDF / DOCX)"]
    end

    subgraph EXTRACT["EXISTING · document_processor.py"]
        TEXT["full extracted text"]
    end

    subgraph FINGERPRINT["NEW · core/paper_fingerprint.py"]
        META["extract title / abstract\n(heuristic: font size, section headers)"]
        QGEN["query_generator\nLLM-assisted: 3-5 search queries\n+ raw title as exact-match query"]
    end

    subgraph CONNECTORS["NEW · connectors/  (SearchConnector interface)"]
        ARXIV["arxiv_connector\narXiv API — free, no key"]
        S2["semantic_scholar_connector\nSemantic Scholar API — free"]
        CROSSREF["crossref_connector\nCrossref API — free, no key"]
        IEEE["ieee_connector\nIEEE Xplore API\n(requires IEEE_API_KEY)"]
        ADS["ads_connector\nNASA ADS API — AAS journals\n(requires ADS_API_TOKEN)"]
        REGISTRY["registry.run_all(queries)\nfan-out via ThreadPoolExecutor\ndedupe by DOI / normalized title"]
    end

    subgraph CACHE["NEW · utils/search_cache.py"]
        DISKCACHE[("on-disk JSON cache\nkeyed by query+connector hash, TTL")]
    end

    subgraph FETCH["NEW · core/candidate_fetcher.py"]
        OA["open-access full text\n(arXiv PDF, OA links)"]
        ABS["abstract-only\n(paywalled candidates)"]
    end

    subgraph SIMENGINE["NEW · core/similarity_engine.py"]
        EMBED["sentence-transformers\n(reuses vector_store embedding model)"]
        CHUNKCMP["chunk-level cosine similarity\ninput chunks × candidate chunks"]
        LEXICAL["TF-IDF / n-gram overlap\n(secondary lexical signal)"]
        BLEND["blended score\n≈70% semantic + 30% lexical"]
    end

    subgraph REPORT["NEW · core/plagiarism_report.py"]
        THRESH["threshold ≥ 60%\n→ flagged as chance of plagiarism"]
        BUILD["structured report:\nflagged[], allCandidates[],\nmatched excerpt pairs, risk level"]
    end

    subgraph API["api.py"]
        EP["POST /api/check-plagiarism"]
    end

    subgraph UI["research_summary_app.html"]
        TOGGLE["'Check for plagiarism' toggle"]
        RESULTS["flagged-source panel\nscore badges, links, excerpt diff"]
    end

    PAPER --> TEXT --> META --> QGEN
    QGEN --> REGISTRY
    REGISTRY <--> DISKCACHE
    REGISTRY --> ARXIV & S2 & CROSSREF & IEEE & ADS
    ARXIV & S2 & CROSSREF & IEEE & ADS --> REGISTRY
    REGISTRY -->|"deduped candidates"| FETCH
    FETCH --> OA
    FETCH --> ABS
    OA --> EMBED
    ABS --> EMBED
    TEXT --> EMBED
    EMBED --> CHUNKCMP --> BLEND
    TEXT -.-> LEXICAL --> BLEND
    BLEND --> THRESH --> BUILD
    BUILD --> EP --> TOGGLE
    TOGGLE --> RESULTS
```

## Request sequence

```mermaid
sequenceDiagram
    participant U as Browser
    participant A as api.py
    participant FP as paper_fingerprint
    participant REG as connectors/registry
    participant SRC as arXiv · S2 · Crossref · IEEE · ADS
    participant CACHE as search_cache
    participant FETCH as candidate_fetcher
    participant SIM as similarity_engine
    participant REP as plagiarism_report

    U->>A: POST /api/check-plagiarism (paper file)
    A->>FP: extract text, title, abstract
    FP->>FP: LLM generates 3-5 search queries
    FP->>REG: run_all(queries)
    loop each connector, in parallel
        REG->>CACHE: lookup(query, connector)
        alt cache hit
            CACHE-->>REG: cached candidates
        else cache miss
            REG->>SRC: search(query)
            SRC-->>REG: candidates (title, abstract, url, doi)
            REG->>CACHE: store(query, connector, candidates)
        end
    end
    REG->>REG: dedupe by DOI / normalized title
    REG-->>A: deduped candidate list
    A->>FETCH: fetch content per candidate
    FETCH-->>A: full text (OA) or abstract-only
    A->>SIM: score(paper_chunks, candidate_chunks)
    SIM->>SIM: embed + cosine similarity (semantic)
    SIM->>SIM: TF-IDF / n-gram overlap (lexical)
    SIM-->>A: blended similarity score per candidate
    A->>REP: build_report(candidates, scores, threshold=0.60)
    REP-->>A: {flagged[], allCandidates[], riskLevel}
    A-->>U: JSON plagiarism report
```

## New modules introduced

| Module | Responsibility |
|---|---|
| `core/paper_fingerprint.py` | Extract title/abstract, generate search queries via LLM |
| `connectors/base.py` | `SearchConnector` interface: `search(query, max_results) -> list[CandidatePaper]` |
| `connectors/arxiv_connector.py` | arXiv API — free, keyless |
| `connectors/semantic_scholar_connector.py` | Semantic Scholar Academic Graph API — free |
| `connectors/crossref_connector.py` | Crossref API — free, keyless, cross-publisher metadata |
| `connectors/ieee_connector.py` | IEEE Xplore API — requires `IEEE_API_KEY` |
| `connectors/ads_connector.py` | NASA ADS API (AAS journals) — requires `ADS_API_TOKEN` |
| `connectors/registry.py` | Fan-out search across connectors, dedupe candidates |
| `utils/search_cache.py` | On-disk TTL cache to limit redundant/rate-limited external calls |
| `core/candidate_fetcher.py` | Resolve full text (open access) or fall back to abstract |
| `core/similarity_engine.py` | Chunk-level embedding similarity + lexical overlap, blended score |
| `core/plagiarism_report.py` | Apply 60% threshold, build the structured report |

## Config additions (`config.py`)

```python
PLAGIARISM_SIMILARITY_THRESHOLD = 0.60   # ≥ this → flagged
SIMILARITY_SEMANTIC_WEIGHT      = 0.70
SIMILARITY_LEXICAL_WEIGHT       = 0.30
SEARCH_MAX_RESULTS_PER_SOURCE   = 10
SEARCH_TIMEOUT_SEC              = 15
SEARCH_CACHE_DIR                = BASE_DIR / "search_cache"
SEARCH_CACHE_TTL_HOURS          = 168     # 1 week

IEEE_API_KEY           = os.getenv("IEEE_API_KEY", "")
ADS_API_TOKEN          = os.getenv("ADS_API_TOKEN", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")  # optional, raises rate limit
```

IEEE and ADS connectors are skipped automatically (with a log warning)
when their key/token isn't configured — the feature still works with
arXiv + Semantic Scholar + Crossref alone.

## Known limitations

- **No Google Scholar coverage** — no official API exists; would require
  SerpAPI (paid, third-party) or scraping (against ToS). Flagged as a
  gap rather than silently worked around.
- **No direct ACM Digital Library connector** — covered indirectly via
  Crossref/Semantic Scholar metadata, which does not include ACM's
  full text.
- **Abstract-only scoring for paywalled candidates** biases the
  similarity score toward abstract/introduction overlap rather than
  full-body text — reported per-candidate via `contentLevel`.
