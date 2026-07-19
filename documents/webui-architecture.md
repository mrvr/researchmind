# Similarity Finder — WebUI Architecture

Extends `research_summary_app.html` with a second mode alongside the
existing "Summarize" view, rather than a separate page — same shell,
same dark theme (`--bg #0d0f14`, `--accent #e8c547`, fonts Syne / DM
Mono / Lora), so it reads as one app, not a bolted-on tool.

## Layout

A segmented mode switcher in the header swaps the main grid between
**Summarize** (existing) and **Similarity Check** (new). The new mode
keeps the established left-input / right-output two-panel pattern.

```mermaid
flowchart TB
    subgraph HEADER["HEADER"]
        SWITCH["mode switcher\n[ Summarize | Similarity Check ]"]
    end

    subgraph LEFT["LEFT PANEL — Paper Under Review"]
        DROP["single-file drop zone\nPDF / DOCX only"]
        F1["input: Paper Title (optional)\noverrides auto-detected title"]
        F2["input: Author(s) (optional)\nhelps disambiguate search"]
        F3["input: Subject / Domain hint (optional)\nscopes generated search queries"]
        SRC["source checkboxes\narXiv · Semantic Scholar · Crossref ·\nIEEE Xplore · NASA ADS\n(IEEE/ADS greyed out if no API key configured)"]
        START["▶ START SIMILARITY CHECK\n(disabled until a file is loaded)"]
    end

    subgraph RIGHT["RIGHT PANEL — Similarity Report"]
        direction TB
        subgraph BANNER["VERDICT BANNER (top of output box)"]
            PCT["highest match %\ne.g. 73%"]
            VERDICT["verdict pill\nORIGINAL / REVIEW SUGGESTED / LIKELY COPY"]
            TITLES["matching paper titles\n(top 3, with source + link)"]
        end
        SUMMARY["Paper Summary\n(general LLM summary of the uploaded paper\n— same summarizer used by Summarize mode)"]
        CANDIDATES["Candidate Matches table\nall searched sources, sortable by score,\nexpandable excerpt comparison per row"]
    end

    SWITCH -.shows.-> LEFT
    SWITCH -.shows.-> RIGHT
    DROP --> START
    F1 & F2 & F3 & SRC --> START
    START -->|"POST /api/check-plagiarism"| BANNER
    BANNER --> SUMMARY --> CANDIDATES
```

## UI states

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Idle: file removed / fields edited
    Idle --> Validating: Start clicked
    Validating --> Idle: no file loaded (blocked, button stays disabled)
    Validating --> Processing: file present
    Processing --> Processing: fingerprint → search → fetch → score
    Processing --> Results: 200 OK
    Processing --> ErrorState: request failed / backend unreachable
    Results --> Idle: Clear clicked
    ErrorState --> Idle: dismissed
    ErrorState --> Processing: Retry clicked
```

## Verdict thresholds shown in the banner

| Highest match score | Pill | Color |
|---|---|---|
| < 40% | `ORIGINAL` | success green |
| 40–59% | `LOW OVERLAP — LIKELY ORIGINAL` | accent2 blue |
| 60–79% | `REVIEW SUGGESTED — POSSIBLE PLAGIARISM` | amber |
| ≥ 80% | `LIKELY COPY — HIGH SIMILARITY` | danger red |

The 60% line matches `PLAGIARISM_SIMILARITY_THRESHOLD` in `config.py` —
the banner surfaces the same number the backend uses to flag a
candidate, it doesn't invent a separate UI-side threshold.

## Request/response contract with the backend

```mermaid
sequenceDiagram
    participant UI as research_summary_app.html
    participant API as api.py

    UI->>API: GET /api/similarity/sources
    API-->>UI: {arxiv: true, semanticScholar: true, crossref: true, ieee: false, ads: false}
    Note over UI: greys out IEEE/ADS checkboxes when a key isn't configured

    UI->>API: POST /api/check-plagiarism<br/>(paper file, title?, authors?, domainHint?, sources[])
    API-->>UI: {status, paperTitle, verdict, highestScore,<br/>flagged[], allCandidates[], summary{topic,overview}}
    Note over UI: verdict banner reads highestScore + verdict + flagged[0..2].title<br/>summary section reads summary.topic / summary.overview<br/>candidates table reads allCandidates[]
```

## New frontend pieces in `research_summary_app.html`

| Element | Purpose |
|---|---|
| `#modeSwitch` | Toggles `.mode-summarize` / `.mode-similarity` sections |
| `#paperDropZone`, `#paperFileInput` | Single-file input for the paper under review |
| `#paperTitleInput`, `#paperAuthorInput`, `#paperDomainInput` | Optional override fields feeding query generation |
| `#sourceChecklist` | Checkboxes populated from `GET /api/similarity/sources` |
| `#startSimilarityBtn` | Disabled until a file is loaded; triggers `runSimilarityCheck()` |
| `#verdictBanner` | Top-of-output block: score, pill, matching titles |
| `#similaritySummary` | General paper summary section |
| `#candidatesTable` | Full candidate list with per-row expand for excerpt comparison |

This reuses the existing `classifyFile`, `formatSize`, `showToast`,
and toast/loading-state CSS rather than duplicating them — the new
mode is additive to the same script, not a separate file.
