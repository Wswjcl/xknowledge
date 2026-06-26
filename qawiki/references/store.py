"""Reference document store (P1 minimal).

Reference documents (Excel rosters, shared tables, web tables) are
preserved **verbatim** — never distilled, never merged. This module
provides the minimal storage primitives the first integration phase
needs:

  * ``register``   — register a new reference doc: assign a doc_id, write
                      a manifest, archive the original file, record it in
                      the source registry with ``source_type=reference``.
  * ``get``        — read a doc's manifest + current version pointer.
  * ``list_docs``  — enumerate registered references.
  * ``update``     — replace a doc's content with a new version (snapshot
                      the old one), used by P5/P6 update flows; here it
                      is a thin version-bump + re-archive.

Structured parsing (rows/schema/conditional query) is deliberately out
of scope for P1 — those land in P2's ``references/query.py``. The
contract here is intentionally schema-light so P2 can extend the
manifest without breaking P1 callers.

Layout::

    knowledge_bank/references/<doc_id>/
        manifest.json       # metadata, versions, state, permissions
        archive/            # original files per version
            v1.xlsx
            v2.xlsx
        current.<ext>       # convenience copy of the live version
"""

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..core.source import hash_file


def _slugify(name: str) -> str:
    """Make a filesystem-safe slug from a filename stem."""
    import re
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-")
    return base or "reference"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ReferenceManifest:
    """In-memory mirror of a reference doc's manifest.json."""

    doc_id: str
    title: str
    source_type: str = "reference"
    origin: str = "xlsx"               # xlsx | csv | html | web | file
    url: Optional[str] = None
    versions: List[Dict[str, Any]] = field(default_factory=list)
    current_version: int = 0           # 1-based; 0 means none yet
    state: str = "pending"             # governance state
    content_hash: str = ""
    ingested_at: str = ""
    last_fetched_at: Optional[str] = None
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)  # P2 schema/rows land here

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReferenceStore:
    """Filesystem-backed store for reference documents."""

    def __init__(self, root: str, registry=None):
        """
        Args:
            root: ``knowledge_bank/references`` directory.
            registry: Optional ``SourceRegistry`` to also record the doc
                in (so it shows up alongside distillable sources). If
                provided, ``register``/``update`` keep it in sync.
        """
        self.root = root
        self.registry = registry
        os.makedirs(self.root, exist_ok=True)

    # ---- paths ----

    def _doc_dir(self, doc_id: str) -> str:
        return os.path.join(self.root, doc_id)

    def _manifest_path(self, doc_id: str) -> str:
        return os.path.join(self._doc_dir(doc_id), "manifest.json")

    def _archive_dir(self, doc_id: str) -> str:
        return os.path.join(self._doc_dir(doc_id), "archive")

    # ---- core ops ----

    def register(
        self,
        file_path: str,
        title: Optional[str] = None,
        doc_id: Optional[str] = None,
        origin: str = "file",
        url: Optional[str] = None,
        state: str = "pending",
        permissions: Optional[Dict[str, List[str]]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ReferenceManifest:
        """Register a new reference document.

        Copies the source file into the archive, writes the manifest,
        and (if a registry is attached) records the source.

        Returns the new manifest. Raises ``FileExistsError`` if ``doc_id``
        already exists.
        """
        doc_id = doc_id or self._derive_doc_id(file_path, title)
        if self.exists(doc_id):
            raise FileExistsError(f"Reference doc {doc_id!r} already exists")

        ext = os.path.splitext(file_path)[1].lower()
        dhash = hash_file(file_path)
        manifest = ReferenceManifest(
            doc_id=doc_id,
            title=title or _slugify(file_path),
            source_type="reference",
            origin=origin,
            url=url,
            current_version=1,
            state=state,
            content_hash=dhash,
            ingested_at=_now(),
            permissions=permissions or {},
            extra=extra or {},
        )
        manifest.versions = [{
            "version": 1,
            "hash": dhash,
            "archived_at": _now(),
            "filename": os.path.basename(file_path),
        }]

        os.makedirs(self._archive_dir(doc_id), exist_ok=True)
        shutil.copy2(file_path, os.path.join(self._archive_dir(doc_id), f"v1{ext}"))
        # Convenience live copy.
        shutil.copy2(file_path, os.path.join(self._doc_dir(doc_id), f"current{ext}"))
        self._write_manifest(manifest)

        if self.registry is not None:
            self.registry.record_sections(doc_id, dhash, {})
            self.registry.set_source_type(doc_id, "reference")
            self.registry.set_state(doc_id, state)
        return manifest

    def update(
        self,
        doc_id: str,
        file_path: str,
        new_state: Optional[str] = None,
    ) -> ReferenceManifest:
        """Replace a reference doc's content with a new version.

        Snapshots the previous version (already archived) and bumps the
        version counter. If ``new_state`` is given the governance state is
        updated too.
        """
        manifest = self.get(doc_id)
        if manifest is None:
            raise KeyError(f"Reference doc {doc_id!r} not found")

        ext = os.path.splitext(file_path)[1].lower()
        dhash = hash_file(file_path)
        # No-op if the content is identical.
        if dhash == manifest.content_hash:
            return manifest

        next_v = manifest.current_version + 1
        manifest.versions.append({
            "version": next_v,
            "hash": dhash,
            "archived_at": _now(),
            "filename": os.path.basename(file_path),
        })
        manifest.current_version = next_v
        manifest.content_hash = dhash
        if new_state:
            manifest.state = new_state
        os.makedirs(self._archive_dir(doc_id), exist_ok=True)
        shutil.copy2(file_path, os.path.join(self._archive_dir(doc_id), f"v{next_v}{ext}"))
        shutil.copy2(file_path, os.path.join(self._doc_dir(doc_id), f"current{ext}"))
        self._write_manifest(manifest)

        if self.registry is not None:
            self.registry.record_sections(doc_id, dhash, {})
            if new_state:
                self.registry.set_state(doc_id, new_state)
        return manifest

    def set_state(self, doc_id: str, state: str):
        """Update only the governance state of a reference doc."""
        manifest = self.get(doc_id)
        if manifest is None:
            raise KeyError(f"Reference doc {doc_id!r} not found")
        manifest.state = state
        self._write_manifest(manifest)
        if self.registry is not None:
            self.registry.set_state(doc_id, state)

    # ---- queries ----

    def exists(self, doc_id: str) -> bool:
        return os.path.exists(self._manifest_path(doc_id))

    def get(self, doc_id: str) -> Optional[ReferenceManifest]:
        path = self._manifest_path(doc_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ReferenceManifest(**data)
        except TypeError:
            # Manifest has unknown keys (P2+); tolerate via extra.
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                known = {
                    "doc_id", "title", "source_type", "origin", "url",
                    "versions", "current_version", "state", "content_hash",
                    "ingested_at", "last_fetched_at", "permissions", "extra",
                }
                extra = {k: v for k, v in data.items() if k not in known}
                data["extra"] = {**data.get("extra", {}), **extra}
                return ReferenceManifest(**{
                    k: v for k, v in data.items() if k in known or k == "extra"
                })
            except Exception:
                return None
        except Exception:
            return None

    def list_docs(self) -> List[ReferenceManifest]:
        out = []
        if not os.path.isdir(self.root):
            return out
        for name in sorted(os.listdir(self.root)):
            if name.startswith("_") or name.startswith("."):
                continue
            m = self.get(name)
            if m is not None:
                out.append(m)
        return out

    def remove(self, doc_id: str):
        """Delete a reference doc entirely (archive + manifest)."""
        d = self._doc_dir(doc_id)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        if self.registry is not None:
            self.registry.remove_doc(doc_id)

    # ---- helpers ----

    def _derive_doc_id(self, file_path: str, title: Optional[str]) -> str:
        base = _slugify(title or file_path)
        candidate = base
        i = 2
        while self.exists(candidate):
            candidate = f"{base}-{i}"
            i += 1
        return candidate

    def _write_manifest(self, manifest: ReferenceManifest):
        os.makedirs(self._doc_dir(manifest.doc_id), exist_ok=True)
        with open(self._manifest_path(manifest.doc_id), "w", encoding="utf-8") as f:
            json.dump(manifest.as_dict(), f, ensure_ascii=False, indent=2)

    # ---- diagnostics ----

    @property
    def status(self) -> Dict[str, Any]:
        docs = self.list_docs()
        return {
            "reference_count": len(docs),
            "published": sum(1 for d in docs if d.state == "published"),
            "pending": sum(1 for d in docs if d.state == "pending"),
            "root": self.root,
        }
