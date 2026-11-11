from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

from datasheet_agent import config


def configure_llama_index() -> None:
    """Configure LlamaIndex to use local Ollama for both LLM and embeddings."""
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        context_window=config.LLM_CONTEXT_WINDOW,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
        json_mode=True,  # ← new: force JSON output from Ollama
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP


def get_extraction_llm() -> Ollama:
    """A fresh LLM instance for structured extraction, with JSON mode on."""
    return Ollama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        context_window=config.LLM_CONTEXT_WINDOW,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
        json_mode=True,
    )