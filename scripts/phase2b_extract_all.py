"""Phase 2b: family-aware extraction over all datasheets."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from datasheet_agent import config
from datasheet_agent.ingest import build_index_from_pdf, load_index
from datasheet_agent.classifier import classify_family
from datasheet_agent.extractor import extract_field
from datasheet_agent.verifier import verify_citation

console = Console()


def process_pdf(pdf_path: Path):
    console.rule(f"[bold cyan]{pdf_path.name}")

    index_name = pdf_path.stem
    persist_dir = config.INDEXES_DIR / index_name
    if persist_dir.exists():
        console.log(f"loading existing index")
        index = load_index(index_name)
    else:
        index = build_index_from_pdf(pdf_path, index_name)

    family = classify_family(index, pdf_path) # family = classify_family(index)
    console.log(f"extracting {len(family.fields)} fields for family [bold]{family.family}[/bold]")

    table = Table(show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_column("Unit")
    table.add_column("Conf.")
    table.add_column("Verified", style="green")
    table.add_column("Citation", overflow="fold", max_width=45)

    verified_count = 0
    for field in family.fields:
        console.log(f"  extracting {field.name}...")
        extraction, nodes = extract_field(index, field)
        verification = verify_citation(extraction, nodes)
        if verification.verified:
            verified_count += 1

        value = extraction.value or "—"
        unit = extraction.unit or ""
        verified = "✓" if verification.verified else "✗"
        citation = extraction.citation_text or "—"
        if len(citation) > 100:
            citation = citation[:100] + "..."

        table.add_row(
            field.display_name, value, unit, extraction.confidence,
            f"{verified} ({verification.match_score:.0%})",
            citation,
        )

    console.print(table)
    console.log(f"[bold]verified: {verified_count}/{len(family.fields)} fields[/bold]")


def main():
    pdfs = sorted(config.DATASHEETS_DIR.glob("*.pdf"))
    if not pdfs:
        console.print("[red]No PDFs in data/datasheets/[/red]")
        return
    console.log(f"processing {len(pdfs)} datasheets")
    for pdf in pdfs:
        process_pdf(pdf)


if __name__ == "__main__":
    main()