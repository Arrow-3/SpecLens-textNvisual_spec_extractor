from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATASHEETS_DIR = DATA_DIR / "datasheets"
INDEXES_DIR = DATA_DIR / "indexes"

# Ollama models
LLM_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# LLM inference
LLM_TEMPERATURE = 0.0
LLM_CONTEXT_WINDOW = 8192
LLM_REQUEST_TIMEOUT = 120

# RAG
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
RETRIEVAL_TOP_K = 5

# Ensure output dirs exist
INDEXES_DIR.mkdir(parents=True, exist_ok=True)