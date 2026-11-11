# SpecLens — Local Datasheet-to-Spec Extraction

A local, retrieval-augmented agent that turns unstructured industrial sensor datasheets into validated, structured JSON specifications. Text and diagrams are both first-class inputs: prose specs flow through a text LLM, while pin assignments, dimensional drawings, and characteristic curves flow through a vision LLM. Every extraction carries a verbatim citation and passes a deterministic verification check before it reaches the output.

Built on the open-source SICK datasheet corpus (five families across LiDAR, photoelectric, time-of-flight, motor-feedback encoders, and flow sensors), running entirely on a developer's own machine through [Ollama](https://ollama.com).

<p align="center">
  <img src="docs/architecture.png" alt="SpecLens architecture overview" width="820"/>
</p>

---

## Motivation for this project

Industrial datasheets are the primary source of truth for sensor integration: every engineer who wires up a new device reads one. They are also stubbornly unstructured. A single PDF carries prose, multi-column tables, pin diagrams rendered as line art, remission curves on log-axis charts, and dimension drawings annotated in two languages. The answer to "what is this sensor's supply voltage?" lives in prose; the answer to "which pin is +Vs?" lives in a labelled diagram; the answer to "what's its sensing range on a 6% remission target?" lives on a plot. A system that can only read text sees roughly 60% of the datasheet.

SpecLens extracts from **both modalities**, then verifies every claim against its own source before emitting it. The pipeline runs offline because customer datasheets are often sensitive, and because reproducibility beats API latency for engineering work.

The design principles:

- **Local-first.** Every model runs through Ollama on consumer hardware. No cloud dependency, no egress of PDF content.
- **Modality-aware routing.** Each schema field declares whether it is text, image, or hybrid. The router sends each field to the model best suited to extract it.
- **Verifiable output.** Every extraction emits a verbatim citation. A deterministic post-check confirms the citation really appears in the source before the field is accepted.
- **Family-aware schemas.** A classifier picks one of five product families and loads that family's field set from YAML. New families are a config change, not a code change.
- **Honest nulls.** The system is tuned to return `null` with a reason, rather than invent a plausible value.

---

## Architecture

The pipeline has four stages. Ingestion and extraction each have parallel text and vision branches that merge into a single validated JSON spec.

<p align="center">
  <img src="docs/pipeline.png" alt="Four-stage pipeline" width="760"/>
</p>

| Stage | Component | Responsibility |
|---|---|---|
| **Ingestion (text)** | PyMuPDF → chunker → `nomic-embed-text` → LlamaIndex | Parses the PDF, splits it into 512-token chunks with 64-token overlap, embeds each chunk into a 768-dim vector, and persists the index to disk. |
| **Ingestion (image)** | PyMuPDF rasteriser → CLIP embed → image store | Renders each page to PNG at ~150 DPI; embeds page thumbnails with CLIP so pages can be retrieved by natural-language queries. |
| **Classification** | Page-1 text + top-k retrieval → Llama 3.1 8B | Reads the datasheet header plus three supporting chunks, selects a product family from vocabulary declared in YAML, and loads that family's field schema. Low-confidence results fall back to a generic schema. |
| **Extraction (text)** | Retrieval (top-5) → Llama 3.1 8B (JSON mode, temp 0) → Pydantic | For each text field, retrieves the most relevant chunks and extracts a value, unit, confidence, and a verbatim citation. |
| **Extraction (image)** | Page retrieval (CLIP) → Qwen2-VL 7B → Pydantic | For each image field (pin assignment, dimensional drawing, working-range plot), retrieves the relevant page and extracts structured data plus a bounding region. |
| **Verification** | Deterministic string match | Normalises whitespace, ellipsis, and case; tries exact substring, falls back to stopword-filtered token overlap at a 0.7 threshold. Images use a crop-back check via a second VLM pass. |

### Why two models, not one

Llama 3.1 8B is a text-only decoder. It cannot see images. Rather than swapping both stages to a weaker multimodal model, SpecLens keeps the strong text model for prose and adds Qwen2-VL 7B specifically for visual fields. The router decides per-field which model to call; the two never compete for the same output. Both run through Ollama and share the same JSON-mode and Pydantic infrastructure.

### Family schemas

Each family is a YAML file in `config/families/`:

| Family | Example device | Field count | Signature fields |
|---|---|---|---|
| `lidar` | picoScan100 | 10 | `scanning_angle`, `angular_resolution`, `number_of_echoes`, `working_range` |
| `photoelectric` | W16 | 10 | `sensing_range`, `light_source`, `switching_frequency`, `output_type` |
| `time_of_flight` | Dx80 | 8 | `sensing_range`, `response_time`, `laser_class`, `interface` |
| `encoder` | ENS/ENM35 | 8 | `resolution`, `revolutions`, `communication_interface`, `operating_speed` |
| `flow` | FTMg | 10 | `measuring_range`, `medium`, `process_pressure`, `process_connection` |
| `generic` | — | 4 | fallback when classification confidence is low |

A field declares its modality:

```yaml
- name: supply_voltage
  display_name: "Supply voltage"
  modality: text
  search_terms: ["supply voltage", "operating voltage", "V DC", "Betriebsspannung"]
  unit: "V"

- name: pin_assignment
  display_name: "Pin assignment"
  modality: image
  search_terms: ["pin assignment", "connector pinout", "M12"]
```

---

## Data contracts

Every stage emits a validated Pydantic DTO. Schemas are versioned per family; a breaking change ships as `families/<name>/v2.yaml` with a migration window.

| DTO | Produced by | Consumed by | Key fields |
|---|---|---|---|
| `IngestedDocument` | Ingestion | Classifier, Extractor | `doc_id`, `path`, `index_path`, `n_chunks`, `n_pages` |
| `FamilyAssignment` | Classifier | Extractor | `family`, `confidence`, `reason`, `fallback` |
| `FieldExtraction` | Extractor (text) | Verifier | `field`, `value`, `unit`, `citation_text`, `confidence`, `notes` |
| `ImageExtraction` | Extractor (vision) | Verifier | `field`, `value`, `page`, `region`, `confidence`, `notes` |
| `VerifiedField` | Verifier | Output assembler | `field`, `value`, `unit`, `verified`, `verification_score`, `citation` |
| `SpecSheet` | Output assembler | API / CLI consumer | `doc_id`, `family`, `fields: list[VerifiedField]`, `extraction_run_id` |

---

## Model stack

| Component | Model | Runtime | VRAM | Role |
|---|---|---|---|---|
| Text embedding | `nomic-embed-text` | Ollama | < 1 GB | Encodes chunks and queries into 768-dim vectors. |
| Image embedding | `clip-vit-b32` | Local Python | < 1 GB | Encodes page thumbnails for text→page retrieval. |
| Text LLM | `llama3.1:8b` (Q4) | Ollama | ~ 5 GB | Classification + text-field extraction, JSON mode, temperature 0. |
| Vision LLM | `qwen2-vl:7b` (Q4) | Ollama | ~ 6 GB | Image-field extraction, structured output. |

Total working set fits on a single 8 GB consumer GPU with model swapping; a 12 GB card runs without swap.

---

## Quickstart

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.com) with `llama3.1:8b`, `qwen2-vl:7b`, and `nomic-embed-text` pulled
- A GPU with ≥ 8 GB VRAM recommended; CPU-only is supported but slow

### Setup

```bash
# 1. Install dependencies
uv sync

# 2. Pull the models
ollama pull llama3.1:8b
ollama pull qwen2-vl:7b
ollama pull nomic-embed-text

# 3. Drop datasheets into data/datasheets/
cp /path/to/your/*.pdf data/datasheets/
```

### Run

```bash
# Extract every datasheet in data/datasheets/
uv run python -m speclens.cli extract-all

# Extract one
uv run python -m speclens.cli extract data/datasheets/PICS150-01000.pdf

# Serve the API
uv run uvicorn speclens.api:app --port 8000
```

Output JSON lands in `data/output/<doc_id>.json`. Indexes persist under `data/indexes/`; re-runs skip embedding.

### API

```http
POST /extract
Content-Type: multipart/form-data

file: <PDF>
```

Returns a `SpecSheet` with every verified field, each carrying its citation and verification score.

---

## Configuration reference

Key settings live in `speclens/config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | `512` | Token size for text chunks. |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks. |
| `TOP_K_TEXT` | `5` | Chunks retrieved per text field. |
| `TOP_K_IMAGE` | `3` | Pages retrieved per image field. |
| `LLM_TEMPERATURE` | `0.0` | Deterministic generation. |
| `LLM_CONTEXT` | `8192` | Context window for Llama. |
| `VLM_CONTEXT` | `32768` | Context window for Qwen2-VL. |
| `CLASSIFIER_CONFIDENCE_THRESHOLD` | `"medium"` | Below this, fall back to generic schema. |
| `VERIFICATION_OVERLAP_THRESHOLD` | `0.7` | Fuzzy-match floor for citation verification. |
| `PAGE_RASTER_DPI` | `150` | Rendering resolution for vision input. |
| `CRITIC_MAX_ATTEMPTS` | `3` | Retries on failed verification (critic loop). |

All environment-configurable via matching `SPECLENS_*` env vars.

---

## Testing

```bash
uv run pytest -v tests
```

The suite covers:

- **DTO round-trips** — every Pydantic schema serialises and deserialises cleanly.
- **Classifier** — page-1 extraction, family vocabulary matching, low-confidence fallback.
- **Verifier** — exact-match path, fuzzy-match path, unicode normalisation (minus signs, non-breaking spaces, smart quotes).
- **Hallucination guards** — adversarial fixtures where the LLM invents a value; verifier must reject.
- **Family loader** — YAML schema validation, duplicate-field detection.

Continuous integration runs the full suite plus lint on every PR via GitHub Actions (`.github/workflows/ci.yml`).

---

## Evaluation

Ground-truth specs for the five reference datasheets are hand-verified from the source PDFs and stored as CSV under `data/ground_truth/`. The eval harness scores each extraction against ground truth in four buckets:

| Bucket | Meaning |
|---|---|
| **Correct** | Extracted value matches ground truth (normalised). |
| **Wrong** | Extracted value present but disagrees with ground truth. |
| **Missed** | Extracted `null`; ground truth has a value. |
| **Correct null** | Extracted `null`; ground truth also empty (e.g. FTMg supply voltage is PoE-only). |

```bash
uv run python -m speclens.eval --family lidar
```

Reports per-field precision and recall, with lenient matching for multi-value fields (IP ratings, switching frequencies across modes).

### Reference results

Measured on the five-datasheet SICK corpus:

| Family | Fields | Precision | Recall | Verified rate |
|---|---|---|---|---|
| `lidar` (picoScan100) | 10 | 0.90 | 0.80 | 0.80 |
| `photoelectric` (W16) | 10 | 0.85 | 0.70 | 0.75 |
| `time_of_flight` (Dx80) | 8 | 0.88 | 0.75 | 0.75 |
| `encoder` (ENM35) | 8 | 0.80 | 0.70 | 0.70 |
| `flow` (FTMg) | 10 | 0.85 | 0.80 | 0.80 |

Classification accuracy across the corpus: 5/5.

---

## Benchmarks

Measured on an RTX 4050 (6 GB) / Ryzen 7 / 16 GB RAM.

| Operation | Duration |
|---|---|
| PDF parse + chunk (20-page datasheet) | 1.5 s |
| Text embedding (2,000 chunks) | 60 s |
| Page rasterisation + CLIP embed (20 pages) | 8 s |
| Classification (per PDF) | 12 s |
| Text field extraction (per field) | 10–14 s |
| Image field extraction (per field) | 18–25 s |
| Full extraction (10 fields, mixed modality) | 2–3 min |
| Index reload (cached) | < 1 s |

GPU memory stays within 6 GB with model swapping between Llama and Qwen2-VL; sustained throughput is a single document at a time on this hardware.

---

## Limitations

Honest caveats, because every pipeline has them.

- **Scanned PDFs are out of scope.** The text branch relies on an embedded text layer. An image-only PDF would need OCR first.
- **Multi-value fields are tension points.** The W16 datasheet declares four switching frequencies depending on mode. The schema asks for one; lenient evaluation reconciles this but it is a design choice, not a solved problem.
- **The verifier confirms citations, not facts.** A correct quote paired with a wrong value passes the deterministic check. The critic LLM catches some of these; a human review catches the rest.
- **Vision verification is weaker than text verification.** Crop-back is a heuristic; a VLM hallucinating on its own crop is harder to detect than a text hallucination.
- **Context window pressure.** Family vocabulary plus top-5 chunks plus prompt approaches Llama's 8k context on dense datasheets. Longer schemas will need a 32k-context model.
- **Single-tenant, single-process.** No concurrency, no queue, no auth. Production deployment would front the API with a worker pool and a job queue.

---

## Roadmap

- **Critic loop.** Second LLM pass that reviews unverified extractions against the source and emits feedback for a bounded retry.
- **Cross-datasheet comparison.** Given several `SpecSheet` outputs, align fields and surface differences (voltage ranges, interface support, mounting).
- **Confidence calibration.** Learn a per-field reliability prior from the eval history.
- **Web UI.** React frontend with side-by-side citation viewer, per-field confidence, and editable verification. The API already shapes its output for this.

---

## References

- SICK AG datasheet corpus — [sick.com](https://www.sick.com)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
- Wang et al., *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World*, 2024
- Nussbaum et al., *Nomic Embed: Training a Reproducible Long Context Text Embedder*, 2024

---

## License

MIT.
