"""Source-type router: declaration-first, LLM-fallback.

Decides whether a new document should be **distilled** (merged into the
insight pool) or kept as a **reference** (preserved verbatim, structured,
versioned). This is the judgment mechanism the architecture calls for.

Resolution order (first hit wins):

  1. Explicit ``declared_type`` argument (CLI ``--type`` / API caller).
  2. Sibling ``manifest.json`` next to the file (``{"source_type": ...}``).
  3. Directory convention: ``…/reference/…`` => reference.
  4. File-extension heuristic (``.xlsx/.csv`` => reference).
  5. LLM classifier on a content snippet + table density (the fallback).
  6. Pure heuristic on content (when no LLM is available): defaults to
     ``distillable`` unless the content is table-heavy.

Each branch is cheap; only step 5 costs an LLM call, and it is reached
only when no declaration exists. Declarations always override the LLM,
so a misclassified doc can be corrected without code changes.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Any

from ..governance.permissions import Role


# Canonical source types.
DISTILLABLE = "distillable"
REFERENCE = "reference"
_VALID = {DISTILLABLE, REFERENCE}

# Directory path segments that imply reference type by convention.
_REFERENCE_SEGMENTS = {"reference", "references", "refs", "shared"}

# Extensions that are almost always reference (tabular) material.
_REFERENCE_EXTS = {".xlsx", ".xls", ".csv", ".tsv"}

# Extensions that are distilled as prose.
_DISTILLABLE_EXTS = {".md", ".markdown", ".docx", ".txt", ".pdf"}


class ClassificationError(Exception):
    pass


@dataclass
class Classification:
    """The router's verdict for one document."""

    source_type: str
    method: str            # how it was decided (declared/manifest/dir/ext/llm/heuristic)
    confidence: float = 1.0
    reason: str = ""
    origin: str = "distillable"   # origin label for provenance (distillable|authored)

    def as_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "method": self.method,
            "confidence": self.confidence,
            "reason": self.reason,
            "origin": self.origin,
        }


def _coerce_type(value: str) -> str:
    value = (value or "").strip().lower()
    if value not in _VALID:
        raise ClassificationError(
            f"source_type must be one of {sorted(_VALID)}, got {value!r}"
        )
    return value


def _read_sibling_manifest(file_path: str) -> Optional[dict]:
    """Look for a manifest.json in the same directory as the file."""
    d = os.path.dirname(os.path.abspath(file_path))
    candidate = os.path.join(d, "manifest.json")
    if os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def _table_density(text: str) -> float:
    """Rough fraction of lines that look like table rows (| a | b |)."""
    if not text:
        return 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    tableish = sum(
        1 for ln in lines
        if ln.count("|") >= 2 or ln.count("\t") >= 2
    )
    return tableish / len(lines)


def _ext_heuristic(file_path: str) -> Optional[Classification]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _REFERENCE_EXTS:
        return Classification(
            source_type=REFERENCE, method="ext", confidence=0.9,
            reason=f"extension {ext} implies tabular reference",
        )
    if ext in _DISTILLABLE_EXTS:
        return Classification(
            source_type=DISTILLABLE, method="ext", confidence=0.7,
            reason=f"extension {ext} typically prose",
        )
    return None


def _dir_convention(file_path: str) -> Optional[Classification]:
    norm = file_path.replace("\\", "/").lower()
    for seg in _REFERENCE_SEGMENTS:
        if f"/{seg}/" in norm:
            return Classification(
                source_type=REFERENCE, method="dir", confidence=0.95,
                reason=f"path under /{seg}/ implies reference",
            )
    return None


def _content_heuristic(snippet: str) -> Classification:
    """Rule-based fallback when no LLM is available."""
    density = _table_density(snippet)
    # Many short tab-separated / pipe-delimited lines => reference.
    if density > 0.5:
        return Classification(
            source_type=REFERENCE, method="heuristic", confidence=0.6,
            reason=f"table density {density:.2f} > 0.5",
        )
    return Classification(
        source_type=DISTILLABLE, method="heuristic", confidence=0.5,
        reason=f"table density {density:.2f} <= 0.5, assume prose",
    )


_LLM_PROMPT = """You are classifying a document for a knowledge base.

Decide whether the document should be:
- "distillable": prose / explanatory content that should be summarized into reusable insights and merged into a shared pool (e.g. technical articles, Q&A, runbooks).
- "reference": tabular / factual / scheduling data that must be preserved verbatim and queried structurally (e.g. staff rosters, Excel schedules, shared tables). NEVER merged.

Document file: {fname}
Table density (fraction of lines that look like rows): {density:.2f}

Content preview (first ~2KB):
---
{preview}
---

Respond with ONLY a JSON object: {{"source_type": "distillable"|"reference", "confidence": 0.0-1.0, "reason": "<short>"}}
"""


def _llm_classify(file_path: str, snippet: str, llm) -> Optional[Classification]:
    """Ask an LLM for the verdict. Returns None on any failure."""
    if llm is None:
        return None
    density = _table_density(snippet)
    prompt = _LLM_PROMPT.format(
        fname=os.path.basename(file_path),
        density=density,
        preview=snippet[:2000],
    )
    try:
        resp = llm.chat(prompt, max_tokens=200)
        # Extract the JSON object from the response.
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            return None
        obj = json.loads(m.group(0))
        st = (obj.get("source_type") or "").strip().lower()
        if st not in _VALID:
            return None
        conf = float(obj.get("confidence", 0.5))
        return Classification(
            source_type=st, method="llm", confidence=conf,
            reason=str(obj.get("reason", ""))[:200],
        )
    except Exception:
        return None


def classify_source(
    file_path: str,
    declared_type: Optional[str] = None,
    manifest: Optional[dict] = None,
    llm: Any = None,
    content_snippet: Optional[str] = None,
) -> Classification:
    """Classify a document's source type.

    Args:
        file_path: Path to the document (used for ext/dir/manifest lookups).
        declared_type: Explicit override (CLI/API). Highest priority.
        manifest: Pre-loaded manifest dict (skips reading sibling file).
        llm: Optional LLM client with a ``.chat(prompt, max_tokens=)`` method
            for the fallback classifier. May be None.
        content_snippet: Optional content preview; if None and needed, the
            classifier reads the file's first bytes itself.

    Returns:
        A Classification describing the verdict and how it was reached.
    """
    # 1. Explicit declaration.
    if declared_type:
        st = _coerce_type(declared_type)
        return Classification(
            source_type=st, method="declared", confidence=1.0,
            reason="explicitly declared by caller",
        )

    # 2. Sibling manifest.json (or provided manifest).
    mf = manifest if manifest is not None else _read_sibling_manifest(file_path)
    if mf and mf.get("source_type"):
        try:
            st = _coerce_type(str(mf["source_type"]))
            return Classification(
                source_type=st, method="manifest", confidence=1.0,
                reason="declared in sibling manifest.json",
            )
        except ClassificationError:
            pass

    # 3. Directory convention.
    c = _dir_convention(file_path)
    if c is not None:
        return c

    # 4. Extension heuristic.
    c = _ext_heuristic(file_path)
    if c is not None:
        return c

    # 5. LLM fallback (needs a content snippet).
    snippet = content_snippet
    if snippet is None:
        snippet = _read_snippet(file_path)
    if snippet:
        c = _llm_classify(file_path, snippet, llm)
        if c is not None:
            return c

    # 6. Content heuristic (no LLM).
    snippet = snippet or ""
    c = _ext_heuristic(file_path)
    if c is not None:
        # Lower-confidence ext guess.
        c.method = "ext+heuristic"
        return c
    return _content_heuristic(snippet)


def _read_snippet(file_path: str, max_bytes: int = 4096) -> str:
    """Read a best-effort text preview of the file (first few KB)."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes)
    except Exception:
        return ""


def origin_for(source_type: str, declared_origin: Optional[str] = None) -> str:
    """Derive a provenance origin label for an item.

    For distillable sources, distinguishes human-authored material from
    automated/trajectory distillation when known. Reference docs always
    carry their own origin in the manifest.
    """
    if declared_origin:
        return declared_origin
    return DISTILLABLE if source_type == DISTILLABLE else REFERENCE
