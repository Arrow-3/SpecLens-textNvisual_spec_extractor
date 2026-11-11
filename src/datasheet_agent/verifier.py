"""Verify that citations actually appear in the retrieved source chunks."""
import re
from dataclasses import dataclass
from llama_index.core.schema import NodeWithScore
from datasheet_agent.schema import FieldExtraction


@dataclass
class VerificationResult:
    verified: bool
    reason: str
    match_score: float  # 0.0 (no overlap) to 1.0 (exact)


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for fuzzy comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _token_overlap(a: str, b: str) -> float:
    """Fraction of a's tokens that appear in b."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def verify_citation(
    extraction: FieldExtraction, nodes: list[NodeWithScore]
) -> VerificationResult:
    """Check whether extraction.citation_text is grounded in the retrieved nodes."""
    if extraction.value is None:
        return VerificationResult(
            verified=True,
            reason="null extraction; no citation to verify",
            match_score=1.0,
        )
    if not extraction.citation_text:
        return VerificationResult(
            verified=False,
            reason="value extracted but no citation provided",
            match_score=0.0,
        )

    cite_norm = _normalize(extraction.citation_text)
    all_context = " ".join(n.node.get_content() for n in nodes)
    context_norm = _normalize(all_context)

    # Exact substring match: strongest evidence.
    if cite_norm in context_norm:
        return VerificationResult(
            verified=True,
            reason="citation appears verbatim in retrieved context",
            match_score=1.0,
        )

    # Fallback: token overlap. Handles cases where the model reformatted
    # slightly (e.g. changed punctuation, expanded abbreviations).
    overlap = _token_overlap(cite_norm, context_norm)
    if overlap >= 0.8:
        return VerificationResult(
            verified=True,
            reason=f"citation matches retrieved context (token overlap {overlap:.0%})",
            match_score=overlap,
        )
    return VerificationResult(
        verified=False,
        reason=f"citation not found in retrieved context (token overlap {overlap:.0%})",
        match_score=overlap,
    )