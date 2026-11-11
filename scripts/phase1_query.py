"""Phase 1 smoke test: index one PDF, ask it three questions."""
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasheet_agent import config
from datasheet_agent.ingest import build_index_from_pdf, load_index

console = Console()

QUESTIONS = [
    "What is the operating voltage range of this sensor?",
    "What is the maximum output power?",
    "What is the maximum switching frequency?",
    "What connection type does this sensor use?",
]


def main():
    pdfs = sorted(config.DATASHEETS_DIR.glob("*.pdf"))
    if not pdfs:
        console.print("[red]No PDFs found in data/datasheets/. Add some first.[/red]")
        return
    pdf = pdfs[0]
    index_name = pdf.stem

    persist_dir = config.INDEXES_DIR / index_name
    if persist_dir.exists():
        console.log(f"loading existing index for {pdf.name}")
        index = load_index(index_name)
    else:
        index = build_index_from_pdf(pdf, index_name)

    engine = index.as_query_engine(similarity_top_k=config.RETRIEVAL_TOP_K)

    for q in QUESTIONS:
        console.print(Panel(q, title="Q", style="cyan"))
        response = engine.query(q)
        console.print(Panel(str(response), title="A", style="green"))
        console.print()


if __name__ == "__main__":
    main()