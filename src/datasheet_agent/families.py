"""Load family YAML specs into typed Python objects."""
from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import BaseModel

from datasheet_agent import config as cfg
from datasheet_agent.schema import FieldSpec


class FamilySpec(BaseModel):
    family: str
    display_name: str
    classifier_hints: list[str]
    fields: list[FieldSpec]


FAMILIES_DIR = cfg.ROOT / "config" / "families"


def load_family(name: str) -> FamilySpec:
    path = FAMILIES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"family spec not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return FamilySpec(**data)


def load_all_families() -> list[FamilySpec]:
    """Load every family YAML in the families directory."""
    return [
        load_family(p.stem)
        for p in sorted(FAMILIES_DIR.glob("*.yaml"))
    ]


def load_non_generic_families() -> list[FamilySpec]:
    """All families except the generic fallback."""
    return [f for f in load_all_families() if f.family != "generic"]