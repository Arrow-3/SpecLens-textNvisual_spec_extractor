Architecture · multimodal extension · 2026-11-09

# Datasheet Extraction · Text + Vision

Text branch keeps Llama 3.1 8B for prose specs. A parallel vision branch routes diagrams, pin assignments and plots through Qwen2-VL 7B. Both merge into one validated JSON spec.

Text LLM **Llama 3.1 8B** Vision LLM **Qwen2-VL 7B** Encoder **nomic-embed-text** Runtime **Ollama · local**

**Text branch**Llama 3.1 8B

**Vision branch**Qwen2-VL 7B

**Encoders**nomic + CLIP

**Router**per-field modality

**Planned**vision verification

### The router

#### Per-field modality routing

Each YAML field declares `modality: text | image | hybrid`. `supply_voltage` stays text. `pin_assignment`, `dimensional_drawing`, `working_range_curve` route to vision. Hybrid fields try text first, fall back to vision on null.

### Text branch

#### Unchanged from current build

PyMuPDF text → nomic-embed → LlamaIndex. All tuning from Phase 2c carries over. Llama 3.1 handles prose specs where it already works.

### Vision branch

#### Qwen2-VL 7B on rasterised pages

PyMuPDF renders each page at \~150 DPI. CLIP embeddings let you query "pin assignment diagram" and retrieve the right page without sending all 20. Qwen2-VL then reads the image and returns structured JSON with a bounding region.

Runs through Ollama · 4-bit quantised fits in 6 GB VRAM.

### Vision verification

#### Crop-back check

String matching doesn't translate. Instead: VLM returns a region, the pipeline crops it, a second VLM pass answers "does this crop support the claim?" Fallback: low-confidence fields flag for human review.

Multimodal extension · after Phase 3 (critic) + Phase 5 (eval) land for text side