# DualGraph → TypeScript + LangGraph Porting Plan

## Overview

Port the DualGraph deep research system from Python to TypeScript, replacing the
imperative main-loop orchestration with a **LangGraph** `StateGraph`.  The Chainlit
web UI (`app.py`) is **not** ported (per issue: "I don't think the ui is necessary").

### Design principles

1. **Benchmark parity** — the issue says "benchmarks should run and be not
   too far off."  This means we must keep the KG-driven chain generation
   logic that is DualGraph's core innovation, including community detection
   and SBM-based chain scoring.  Where Python uses `igraph` + `leidenalg`
   we substitute `graphology` + `graphology-communities-louvain` (Louvain
   is algorithmically close to Leiden).  Cosine similarity for
   entity-concept scoring is trivial to implement without numpy.
2. **LangChain for LLM calls, LangGraph for orchestration** — LLM calls go through
   `@langchain/openai` (`ChatOpenAI` for chat, `OpenAIEmbeddings` for embeddings),
   preserving the same prompts and retry logic.  LangGraph is used
   to model the init → iterate → terminate → write control flow.
3. **Minimal new dependencies** — only add packages that replace
   Python-only libraries.  Reuse the same YAML prompt templates verbatim.

## Source → Target mapping

| Python source                         | TypeScript target            | Notes                                            |
|---------------------------------------|------------------------------|--------------------------------------------------|
| `data_model.py`                       | `src/dataModel.ts`           | Interfaces + pure helper functions               |
| `llm_utils.py`                        | `src/llmUtils.ts`            | `@langchain/openai` (ChatOpenAI, OpenAIEmbeddings); retry logic |
| `utils_module.py`                     | `src/utils.ts`               | Usage tracking, JSON repair, file helpers         |
| `outline_module.py`                   | `src/outlineModule.ts`       | Create / update outline, generate search queries  |
| `knowledge_graph_module.py`           | `src/knowledgeGraphModule.ts`| KG create/update, chain generation, community det.|
| `search_module.py`                    | `src/searchModule.ts`        | Bing / Serper search, URL fetch + summarize       |
| `write_module.py`                     | `src/writeModule.ts`         | Section-by-section report writing, ref dedup      |
| `terminal_module.py`                  | `src/terminalModule.ts`      | Multi-criteria early stopping                     |
| `main.py` (pipeline orchestration)    | `src/graph.ts`               | **LangGraph StateGraph** replaces imperative loop |
| `main.py` (CLI + dataset loading)     | `src/main.ts`                | Commander CLI, dataset JSONL loader               |
| `log_tee.py`                          | *(not ported)*               | Node `console` logging is sufficient              |
| `neo4j_kg_store.py`                   | *(not ported)*               | Optional Neo4j; can be added later                |
| `app.py`                              | *(not ported)*               | UI not needed per issue                           |
| `prompt_lib/*.yaml`                   | `src/prompt_lib/*.yaml`      | Copied as-is (YAML prompts are language-agnostic) |

## Architecture: LangGraph state graph

The Python `process_single_report_og_kg()` uses a sequential init-then-loop
pattern.  We model this as a LangGraph `StateGraph` with **three** high-level
nodes — keeping the graph simple and easy to reason about:

```
          ┌──────────────────────────────┐
          │         initialize           │  (iter 0)
          │  • createOutline             │
          │  • generateOgQueries         │
          │  • executeSearch             │
          │  • buildKg                   │
          │  • generateKgQueries         │
          └──────────┬───────────────────┘
                     ▼
          ┌──────────────────────────────┐
          │         iterate              │◄──────┐
          │  • executeSearch             │       │
          │  • updateKg                  │       │
          │  • updateOutline (by KG)     │       │
          │  • planNextQueries           │       │
          │  • judgeTerminal             │       │
          │  • save intermediate files   │       │
          └──────────┬───────────────────┘       │
                     │                           │
                     ├── continue ───────────────┘
                     │
                     ▼ (stop OR maxIter reached)
          ┌──────────────────────────────┐
          │       writeReport            │
          │  • write sections via LLM    │
          │  • dedup & renumber refs     │
          │  • save report + evidence    │
          └──────────────────────────────┘
```

**Why three nodes instead of many?**  The init steps and the iteration
sub-steps have no branching between them — they are always sequential.
Splitting them into separate graph nodes adds boilerplate without enabling
any LangGraph feature (parallelism, conditional routing).  The only
meaningful branching point is the continue-or-stop decision after each
iteration, which is a single conditional edge.

### State definition

LangGraph JS uses `Annotation.Root` for state.  We must avoid
non-serializable types (`Set`, `Map`) in the state since LangGraph may
serialize state for checkpointing.  Use plain arrays/objects instead:

```typescript
import { Annotation } from "@langchain/langgraph";

const ResearchState = Annotation.Root({
  // ── Config (set once at invocation, never mutated by nodes) ──
  rootQuery:            Annotation<string>,
  reportId:             Annotation<number>,
  language:             Annotation<string>,
  maxIter:              Annotation<number>,
  usageFile:            Annotation<string>,
  reportDir:            Annotation<string>,
  productDir:           Annotation<string>,
  kgQueryNum:           Annotation<number>,
  ogQueryNum:           Annotation<number>,
  searchProvider:       Annotation<string>,
  disableEarlyStopping: Annotation<boolean>,

  // ── Evolving state (mutated by graph nodes) ──
  outline:              Annotation<string>,
  knowledgeGraph:       Annotation<KnowledgeGraphJson>,  // serializable snapshot
  searchQueries:        Annotation<string[]>,
  historySearchQueries: Annotation<string[]>,
  visitedUrls:          Annotation<string[]>,             // not Set — serializable
  visitedEdges:         Annotation<string[]>,             // "srcId:tgtId:type" strings
  iteration:            Annotation<number>,
  isTerminal:           Annotation<boolean>,

  // ── Output ──
  report:               Annotation<string>,
});
```

`KnowledgeGraphJson` is a plain-object snapshot of the KG (obtained via
`JSON.parse(JSON.stringify(kg))` after converting Maps to objects).  Inside
each node function, we rehydrate it into the richer in-memory
`KnowledgeGraph` type, operate on it, then dehydrate before returning.
This keeps the graph state fully serializable.

### LLM integration approach

LLM calls go through `@langchain/openai`'s `ChatOpenAI` (wrapped in the
`LLMModel` class in `llmUtils.ts`), with `OpenAIEmbeddings` for vector
embeddings.  This ensures:
- Full LangChain integration for both orchestration and LLM calls.
- Identical prompt formatting and retry logic to the Python original.
- Easy to compare token usage between Python and TS runs.

LangGraph is used for control flow (the `StateGraph`) and LangChain handles
invocation.

## Step-by-step porting order

### Phase 1 — Project scaffolding
1. Create `DualGraph/ts/` with `package.json`, `tsconfig.json`.
2. Install dependencies (see Dependencies section below).
3. Copy `prompt_lib/*.yaml` into `src/prompt_lib/`.
4. Add `node_modules/` and `dist/` to `.gitignore`.

### Phase 2 — Core modules (unit-testable without LLM)
5. `src/dataModel.ts` — interfaces, KG helpers, merge/cluster logic,
   text/JSON serialization, serialization/deserialization helpers.
6. `src/utils.ts` — usage tracking, JSON repair, file helpers, dedup.
7. `src/llmUtils.ts` — LangChain ChatOpenAI wrapper, retry with backoff, embeddings.

### Phase 3 — Domain modules
8. `src/outlineModule.ts` — create/update outline, generate search queries.
9. `src/knowledgeGraphModule.ts` — KG create/update, chain generation:
   - **Louvain community detection** via `graphology` +
     `graphology-communities-louvain` (replaces Python's Leiden).
   - **SBM probability/entropy chain scoring** — ported as-is (pure math,
     no external deps needed).
   - **Cosine similarity** for entity-concept scoring — manual dot-product
     implementation (replaces numpy).
   - **Enrich chains** — ported as-is.
   - **Cross-community chains** — ported as-is using community info.
   - **HDBSCAN semantic clustering** — omitted (no mature JS equivalent).
     The Louvain community detection serves a similar role for the prompts.
10. `src/searchModule.ts` — Bing/Serper HTTP search, Jina/Firecrawl page
    reading, LLM URL selection + evidence summarization.  Use Node `fetch`.
11. `src/writeModule.ts` — section-by-section report writing, reference
    dedup/renumber.
12. `src/terminalModule.ts` — multi-criteria early stopping (6 dimensions).

### Phase 4 — LangGraph orchestration + CLI
13. `src/graph.ts` — LangGraph `StateGraph` with `initialize`, `iterate`,
    `writeReport` nodes and the conditional continue/stop edge.
14. `src/main.ts` — Commander CLI, `.env` loading, dataset JSONL parsing,
    output directory setup, invoke graph per report (sequential by default,
    with optional `--max-concurrency` using p-limit for multi-report runs).

### Phase 5 — Testing & validation
15. Unit tests for `dataModel` (KG construction, merge, serialization,
    text/JSON output, cluster map).
16. Unit tests for `utils` (JSON repair, dedup, atomic write).
17. Unit tests for `writeModule` (reference dedup/renumber).
18. Unit tests for `graph` (mock LLM, verify state transitions through
    the full init → iterate → write flow).
19. Build verification (`npm run build` — zero tsc errors).
20. Integration smoke test (requires `.env` with API keys; run the example
    dataset and compare output against the Python baseline).

## Feature parity analysis

### Kept (critical for benchmarks)

| Feature                              | Python implementation         | TypeScript approach                              |
|--------------------------------------|-------------------------------|--------------------------------------------------|
| KG-driven chain generation           | Full pipeline                 | Ported as-is                                     |
| Community detection                  | `igraph` + `leidenalg`        | `graphology` + `graphology-communities-louvain`   |
| SBM probability/entropy scoring      | Pure math + community info    | Ported as-is (no external deps)                  |
| Cosine similarity (entity-concept)   | `numpy`                       | Manual dot-product (trivial)                     |
| Enrich chains (low-evidence edges)   | Direct KG inspection          | Ported as-is                                     |
| Cross-community exploration chains   | Bridge/hub node selection      | Ported as-is                                     |
| Multi-criteria early stopping        | 6-dimension LLM judgment      | Ported as-is                                     |
| All 13 YAML prompt templates         | YAML files                    | Copied verbatim                                  |
| Bing + Serper search                 | `requests`                    | Node `fetch`                                     |
| Jina / Firecrawl page reading        | `requests`                    | Node `fetch`                                     |
| Usage/token tracking                 | JSON file + thread lock       | JSON file (no lock needed — single-threaded)     |
| Intermediate artifact saving         | Per-iteration file writes     | Same file writes in graph nodes                  |
| Reference dedup/renumber             | Regex-based                   | Ported as-is                                     |

### Omitted (low impact on benchmarks)

| Feature                              | Reason                                                            |
|--------------------------------------|-------------------------------------------------------------------|
| HDBSCAN semantic clustering          | No mature JS equivalent; Louvain community detection covers the   |
|                                      | same conceptual role for prompts.  Benchmark impact: minimal      |
|                                      | (Python also makes this optional via env var).                    |
| Neo4j KG store                       | Optional persistence layer; in-memory KG is the primary path.    |
| Chainlit web UI                      | Per issue: "I don't think the ui is necessary."                   |
| pyvis visualization                  | Developer tool; not part of the benchmark pipeline.              |
| Azure AD authentication              | Enterprise auth; API-key auth covers all benchmark scenarios.    |
| TOON format (pytoony)                | Niche serialization format; JSON representation is used instead. |
| Concurrent multi-report processing   | Python uses ThreadPoolExecutor.  TS version processes reports     |
|                                      | sequentially by default.  Can add p-limit concurrency later.     |

## Dependencies

```json
{
  "@langchain/core": "^0.3.0",
  "@langchain/langgraph": "^0.2.0",
  "commander": "^13.0.0",
  "dotenv": "^16.4.0",
  "graphology": "^0.25.0",
  "graphology-communities-louvain": "^2.0.0",
  "graphology-types": "^0.24.0",
  "js-yaml": "^4.1.0",
  "jsonrepair": "^3.12.0",
  "graphology": "^0.25.0",
  "graphology-communities-louvain": "^2.0.0",
  "graphology-types": "^0.24.0",
  "js-yaml": "^4.1.0",
  "jsonrepair": "^3.12.0",
  "zod": "^3.24.0"
}
```

Note: `@langchain/openai` provides both `ChatOpenAI` and `OpenAIEmbeddings`.
The raw `openai` SDK is no longer a direct dependency — it is pulled in
transitively by `@langchain/openai`.

## Risks and mitigations

| Risk                                          | Mitigation                                              |
|-----------------------------------------------|---------------------------------------------------------|
| Louvain ≠ Leiden — slightly different results  | Louvain is the predecessor of Leiden; both produce       |
|                                               | similar community partitions.  Benchmark impact is low. |
| No HDBSCAN → fewer semantic clusters in prompts| Python makes HDBSCAN optional via env var.  The LLM     |
|                                               | prompts still work without cluster annotations.         |
| LangGraph state serialization with complex KG  | Serialize KG to JSON before returning from nodes;       |
|                                               | deserialize at node entry.  Avoids Map/Set issues.      |
| Single-threaded Node.js vs Python threads      | For benchmarks, only one report runs at a time anyway.  |
|                                               | Multi-report parallelism can use Promise.all + p-limit. |
| Prompt template differences (string formatting)| YAML templates are copied verbatim.  Template vars use  |
|                                               | the same string interpolation as Python.                |
| Error handling / partial saves                 | Each graph node wraps its work in try/catch and saves   |
|                                               | intermediate artifacts, matching the Python behavior.   |

## Testing strategy

- **Unit tests** (vitest, offline — no API keys):
  - Data model: KG construction, node/edge CRUD, merge, serialization,
    text/JSON output, cluster map.
  - Utils: JSON repair, dedup, atomic write.
  - Write module: reference dedup/renumber.
  - Graph: mock `LLMModel` that returns canned responses → verify full
    init → iterate → write state transitions.
- **Build test**: `tsc` compiles with zero errors.
- **Integration / benchmark** (requires `.env` with API keys):
  - Run the `example` dataset through both Python and TS.
  - Compare: (a) token usage, (b) number of iterations, (c) output report
    structure and reference count.
  - The issue's acceptance criterion is "not too far off" — we target
    ±10% token usage and comparable report quality.
