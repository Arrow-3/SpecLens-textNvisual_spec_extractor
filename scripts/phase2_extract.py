"""Phase 2: extract structured fields from an indexed SICK datasheet."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from datasheet_agent import config
from datasheet_agent.ingest import build_index_from_pdf, load_index
from datasheet_agent.extractor import extract_field
from datasheet_agent.verifier import verify_citation
from datasheet_agent.schema import SICK_PHOTOELECTRIC_FIELDS

console = Console()


def main():
    pdfs = sorted(config.DATASHEETS_DIR.glob("*.pdf"))
    if not pdfs:
        console.print("[red]No PDFs in data/datasheets/[/red]")
        return
    pdf = pdfs[0]
    console.rule(f"[bold cyan]{pdf.name}")

    index_name = pdf.stem
    persist_dir = config.INDEXES_DIR / index_name
    if persist_dir.exists():
        console.log(f"loading existing index for {pdf.name}")
        index = load_index(index_name)
    else:
        index = build_index_from_pdf(pdf, index_name)

    table = Table(show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_column("Unit")
    table.add_column("Conf.")
    table.add_column("Verified", style="green")
    table.add_column("Citation", overflow="fold", max_width=45)

    for field in SICK_PHOTOELECTRIC_FIELDS:
        console.log(f"extracting {field.name}...")
        extraction, nodes = extract_field(index, field)
        verification = verify_citation(extraction, nodes)

        value = extraction.value or "—"
        unit = extraction.unit or ""
        conf = extraction.confidence
        verified = "✓" if verification.verified else "✗"
        citation = extraction.citation_text or "—"
        if len(citation) > 100:
            citation = citation[:100] + "..."

        table.add_row(
            field.display_name, value, unit, conf,
            f"{verified} ({verification.match_score:.0%})",
            citation,
        )

    console.print(table)


if __name__ == "__main__":
    main()