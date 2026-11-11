"""Extract structured fields from an indexed datasheet."""
from __future__ import annotations
import json
from typing import Optional
from rich.console import Console

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from datasheet_agent import config
from datasheet_agent.llm import get_extraction_llm
from datasheet_agent.schema import FieldExtraction, FieldSpec

console = Console()


EXTRACTION_PROMPT_TEMPLATE = """You are extracting technical specifications from an industrial sensor datasheet.

Your task: extract the value of "{display_name}" from the excerpts below.

CRITICAL RULES:
1. Only use information EXPLICITLY present in the excerpts. Do not use general knowledge about similar sensors.
2. If the excerpts do not contain this field, you MUST return null. Do not guess. Do not fill in a plausible value.
3. The citation_text MUST be an EXACT COPY of text from the excerpts. Copy the exact characters as they appear. Do NOT rewrite as a sentence. Do NOT paraphrase.
4. If the value appears in a table cell (e.g. "Supply voltage: 12 V ... 30 V"), the citation must be exactly that text — not "The supply voltage is 12 to 30 V".

Example of CORRECT citation:
  Excerpt: "Supply voltage: 10 ... 30 V DC"
  Citation: "Supply voltage: 10 ... 30 V DC"

Example of WRONG citation (paraphrasing — this is forbidden):
  Excerpt: "Supply voltage: 10 ... 30 V DC"
  Citation: "The supply voltage is 10 to 30 V DC."

Example of WRONG extraction (hallucination — this is forbidden):
  Excerpts contain no mention of RS-485.
  Extraction: {{"value": "RS-485", ...}}
  Correct: {{"value": null, "citation_text": null, "confidence": "low", "notes": "not found in provided excerpts"}}

Return only valid JSON matching this schema:
{{
  "value": string or null,
  "unit": string or null,
  "citation_text": string or null,
  "confidence": "high" | "medium" | "low",
  "notes": string or null
}}

Excerpts from the datasheet:
---
{context}
---

Extract "{display_name}" from the excerpts above. If not present, return null. Return only the JSON object, no explanation."""


def _retrieve_chunks(
    index: VectorStoreIndex, field: FieldSpec, top_k: int = 5
) -> list[NodeWithScore]:
    """Retrieve the most relevant chunks for a field's query."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(field.query)


def _format_context(nodes: list[NodeWithScore]) -> str:
    """Concatenate retrieved chunks with clear separators."""
    parts = []
    for i, node in enumerate(nodes, 1):
        text = node.node.get_content().strip()
        parts.append(f"[Excerpt {i}] {text}")
    return "\n\n".join(parts)


def extract_field(
    index: VectorStoreIndex,
    field: FieldSpec,
    top_k: int = 5,
) -> tuple[FieldExtraction, list[NodeWithScore]]:
    """Extract a single field, returning the extraction and the retrieved chunks."""
    nodes = _retrieve_chunks(index, field, top_k=top_k)
    context = _format_context(nodes)
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        display_name=field.display_name,
        context=context,
    )

    llm = get_extraction_llm()
    response = llm.complete(prompt)
    raw = response.text.strip()

    try:
        data = json.loads(raw)
        extraction = FieldExtraction(**data)
    except (json.JSONDecodeError, ValueError) as e:
        console.log(f"[red]parse error for {field.name}: {e}[/red]")
        console.log(f"[dim]raw response: {raw[:300]}[/dim]")
        extraction = FieldExtraction(
            value=None,
            citation_text=None,
            confidence="low",
            notes=f"JSON parse error: {e}",
        )

    return extraction, nodes