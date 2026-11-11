from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from rich.console import Console

from datasheet_agent import config
from datasheet_agent.llm import configure_llama_index

console = Console()


def build_index_from_pdf(pdf_path: Path, index_name: str) -> VectorStoreIndex:
    """Parse a PDF, chunk it, embed the chunks, and persist the index."""
    configure_llama_index()
    console.log(f"parsing {pdf_path.name}")

    documents = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
    console.log(f"loaded {len(documents)} document sections")

    console.log("building index (this triggers embeddings for every chunk)")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    persist_dir = config.INDEXES_DIR / index_name
    index.storage_context.persist(persist_dir=str(persist_dir))
    console.log(f"persisted to {persist_dir}")
    return index


def load_index(index_name: str) -> VectorStoreIndex:
    """Load a previously persisted index from disk."""
    configure_llama_index()
    persist_dir = config.INDEXES_DIR / index_name
    storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage)