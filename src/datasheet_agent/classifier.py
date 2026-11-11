"""LLM-based family classifier using the first-page header of a datasheet."""
from __future__ import annotations
import json
from pathlib import Path
import pymupdf
from rich.console import Console

from llama_index.core import VectorStoreIndex

from datasheet_agent.families import FamilySpec, load_non_generic_families, load_family
from datasheet_agent.llm import get_extraction_llm

console = Console()


CLASSIFIER_PROMPT = """You are classifying an industrial sensor datasheet into one product family.

Product families and their characteristic vocabulary:
{families}

The following is the first page (or two) of the datasheet — this is where product family is declared:
---
{page1_text}
---

Additional excerpts from the datasheet body:
---
{context}
---

Which family does this datasheet belong to?

Rules:
- Match the family name that best fits the datasheet's declared product type.
- Look for exact terms from each family's vocabulary appearing in the datasheet text.
- If the datasheet does not clearly fit any family, use "generic".
- Do not hallucinate. If the text does not mention lidar terms, do not classify as lidar.

Respond with valid JSON only:
{{"family": "<one of: {family_names}>", "confidence": "high" | "medium" | "low", "reason": "which specific terms in the text led to this choice"}}"""


def _summarize_families(families: list[FamilySpec]) -> str:
    """List each family with its display name and vocabulary examples."""
    lines = []
    for f in families:
        # Take up to 6 characteristic terms across all field search_terms
        terms: list[str] = []
        for field in f.fields:
            terms.extend(field.search_terms[:2])
        # Also include classifier_hints
        vocab = list(dict.fromkeys(f.classifier_hints + terms))[:10]
        lines.append(f"- {f.family} ({f.display_name}): vocabulary = {', '.join(vocab)}")
    return "\n".join(lines)


def _extract_first_pages(pdf_path: Path, n_pages: int = 2) -> str:
    """Extract raw text from the first N pages of a PDF."""
    try:
        doc = pymupdf.open(str(pdf_path))
        pages_to_read = min(n_pages, doc.page_count)
        text_parts = []
        for i in range(pages_to_read):
            page_text = doc[i].get_text().strip()
            if page_text:
                text_parts.append(page_text)
        doc.close()
        return "\n\n---PAGE BREAK---\n\n".join(text_parts)
    except Exception as e:
        console.log(f"[yellow]could not extract first pages: {e}[/yellow]")
        return ""


def classify_family(
    index: VectorStoreIndex,
    pdf_path: Path,
    top_k: int = 3,
) -> FamilySpec:
    """Classify a datasheet by combining page-1 text with retrieval."""
    families = load_non_generic_families()
    family_names = " | ".join(f.family for f in families)

    # Page 1 is where SICK declares the product family in the header.
    page1_text = _extract_first_pages(pdf_path, n_pages=2)

    # Also retrieve a couple of characteristic chunks for supporting context.
    query = "measurement principle features product type"
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    context = "\n\n".join(n.node.get_content()[:600] for n in nodes)

    prompt = CLASSIFIER_PROMPT.format(
        families=_summarize_families(families),
        family_names=family_names,
        page1_text=page1_text[:2000],
        context=context,
    )

    llm = get_extraction_llm()
    response = llm.complete(prompt)
    raw = response.text.strip()

    try:
        data = json.loads(raw)
        family_name = data.get("family", "generic")
        confidence = data.get("confidence", "low")
        reason = data.get("reason", "")
    except json.JSONDecodeError:
        console.log(f"[yellow]classifier parse error; raw: {raw[:200]}[/yellow]")
        family_name, confidence, reason = "generic", "low", "parse error"

    console.log(f"classified as [bold cyan]{family_name}[/bold cyan] ({confidence}) — {reason}")

    if confidence == "low":
        console.log("[yellow]low confidence → falling back to generic spec[/yellow]")
        return load_family("generic")

    try:
        return load_family(family_name)
    except FileNotFoundError:
        console.log(f"[yellow]unknown family '{family_name}' → generic[/yellow]")
        return load_family("generic")