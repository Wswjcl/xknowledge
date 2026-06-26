"""Utility functions - image encoding, insight library I/O.
Adapted from XSkill eval/exskill/experience_utils.py
Original work by Jiang et al. (ICML 2026, MIT License)

The insight library is persisted with rich per-insight metadata:
    {"insights": {"E0": {"text": ..., "source_doc": ..., "section": ...,
                         "added_at": ..., "content_hash": ...}, ...}}

Old libraries ({id: "<plain text>"}) are auto-wrapped on load for
backward compatibility. Callers that only need the text view (the
distillation/retrieval core) keep using the Dict[str, str] helpers, so
the new schema is isolated behind the *_insight_meta helpers below.
"""

import os
import json
import base64
import io
from typing import Dict, Any
from PIL import Image
from ..prompts.query import INSIGHT_INJECTION_HEADER


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI.

    Args:
        image: PIL Image to convert

    Returns:
        Base64-encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --------- Experience Library I/O ---------

def _wrap_entry(value: Any) -> Dict[str, Any]:
    """Normalize one library entry into the metadata dict form.

    Accepts both the legacy plain-string schema and the new metadata
    object schema. Always returns a dict with at least a ``text`` key.
    """
    if isinstance(value, dict):
        entry = dict(value)
        if "text" not in entry:
            # Dict without explicit text - reconstruct from first str field
            for v in entry.values():
                if isinstance(v, str):
                    entry["text"] = v
                    break
        entry.setdefault("text", "")
        return entry
    if isinstance(value, str):
        return {"text": value}
    return {"text": str(value) if value is not None else ""}


def load_insight_library(path: str) -> Dict[str, str]:
    """Load insights from a JSON file as a flat id -> text mapping.

    This is the text-only view consumed by the distillation/retrieval
    core. Use ``load_insight_library_meta`` to also get the per-insight
    metadata (source_doc, section, ...).

    Args:
        path: Path to the JSON file

    Returns:
        Dictionary mapping insight IDs to insight text
    """
    meta = load_insight_library_meta(path)
    return {eid: entry.get("text", "") for eid, entry in meta.items()}


def load_insight_library_meta(path: str) -> Dict[str, Dict[str, Any]]:
    """Load insights together with their per-entry metadata.

    Legacy plain-string entries are wrapped into {"text": str} so callers
    always see the new schema regardless of what is on disk.

    Args:
        path: Path to the JSON file

    Returns:
        Dictionary mapping insight IDs to metadata dicts. Each dict
        contains at least ``text``; other keys (source_doc, section,
        added_at, content_hash) are present only when previously stored.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if isinstance(data, dict) and "insights" in data:
        data = data["insights"]
    if not isinstance(data, dict):
        return {}
    return {eid: _wrap_entry(val) for eid, val in data.items()}


def save_insight_library(
    path: str,
    experiences: Dict[str, str],
    meta: Dict[str, Dict[str, Any]] = None,
):
    """Save insights to a JSON file.

    When ``meta`` is provided (mapping id -> metadata dict), the on-disk
    schema is {id: {text, ...meta...}}; the ``experiences`` text values
    win over any stale ``text`` key inside ``meta``. When ``meta`` is
    omitted, a plain {id: text} document is written (legacy schema).

    Args:
        path: Path to save the JSON file
        experiences: Dictionary mapping insight IDs to insight text
        meta: Optional per-id metadata to persist alongside the text
    """
    if meta:
        merged = {}
        for eid, text in experiences.items():
            entry = dict(meta.get(eid, {}))
            entry["text"] = text
            merged[eid] = entry
        out = merged
    else:
        out = dict(experiences)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"insights": out}, f, ensure_ascii=False, indent=2)


save_library = save_insight_library
load_existing = load_insight_library


def load_insights(path: str) -> Dict[str, str]:
    """Load insights from a JSON file (text view). Alias of load_insight_library."""
    return load_insight_library(path)


load_experiences = load_insights


def format_insights_for_prompt(experiences: Dict[str, str], max_items: int = 32) -> str:
    """Format insights for injection into prompts.

    Args:
        experiences: Dictionary mapping insight IDs to insight text
        max_items: Maximum number of insights to include

    Returns:
        Formatted string for prompt injection
    """
    if not experiences:
        return ""
    items = list(experiences.items())[:max_items]
    bullets = "\n".join([f"- [{k}] {v}" for k, v in items])
    return INSIGHT_INJECTION_HEADER.format(bullets=bullets)


format_for_prompt = format_insights_for_prompt
