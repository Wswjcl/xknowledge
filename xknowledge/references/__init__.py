"""Reference document storage (preserve-verbatim sources).

Reference docs (Excel rosters, shared tables, web tables) are kept as-is
rather than distilled. P1 ships the storage primitives; structured
parsing and conditional query land in P2.
"""

from .store import ReferenceStore, ReferenceManifest

__all__ = ["ReferenceStore", "ReferenceManifest"]
