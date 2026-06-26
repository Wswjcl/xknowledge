"""Review queue: the holding area for ``pending`` knowledge items.

This is the concrete mechanism behind the governance rule "automatically
distilled content is published only after review." Every time the
distillation pipeline (or a contributor submission) produces an item in
the ``pending`` state, it is appended here. Reviewers then work through
the queue via ``approve`` / ``reject``.

The queue is a single JSON file (``review_queue.json``) under the KB:

    {
      "items": [
        {
          "id": "rv-0001",
          "target_type": "insight",        # insight | reference | framework
          "target_id": "E7",               # insight id / doc_id / "global"
          "title": "...",
          "reason": "auto-distilled from runbook.md",
          "proposed_by": "system",
          "state": "pending",
          "created_at": "...",
          "decided_at": null,
          "decided_by": null,
          "decision": null                  # approved | rejected
          "payload_preview": "..."          # short text for review UI
        }
      ]
    }

Decided items (approved/rejected) are kept for audit and filtered out of
the default "open" listing.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


# Target type discriminator for review items.
TARGET_INSIGHT = "insight"
TARGET_REFERENCE = "reference"
TARGET_FRAMEWORK = "framework"
_VALID_TARGETS = {TARGET_INSIGHT, TARGET_REFERENCE, TARGET_FRAMEWORK}

STATE_OPEN = "pending"
STATE_APPROVED = "approved"
STATE_REJECTED = "rejected"


@dataclass
class ReviewItem:
    """One pending item awaiting a review decision."""

    id: str
    target_type: str
    target_id: str
    title: str = ""
    reason: str = ""
    proposed_by: str = "system"
    state: str = STATE_OPEN
    created_at: str = ""
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None
    decision: Optional[str] = None
    payload_preview: str = ""


class ReviewQueue:
    """Persistent review queue backed by a JSON file."""

    def __init__(self, path: str):
        self.path = path
        self._items: List[ReviewItem] = []
        self._load()

    # ---- persistence ----

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            self._items = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("items", []) if isinstance(data, dict) else []
            self._items = [ReviewItem(**r) for r in raw if isinstance(r, dict)]
        except Exception:
            self._items = []

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {"items": [asdict(i) for i in self._items]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- mutation ----

    def submit(
        self,
        target_type: str,
        target_id: str,
        title: str = "",
        reason: str = "",
        proposed_by: str = "system",
        payload_preview: str = "",
    ) -> ReviewItem:
        """Register a new pending item.

        Returns the created ReviewItem. Idempotent on (target_type,
        target_id): if an OPEN item already exists for that target, it is
        returned unchanged rather than duplicated.
        """
        if target_type not in _VALID_TARGETS:
            raise ValueError(
                f"target_type must be one of {_VALID_TARGETS}, got {target_type!r}"
            )
        existing = self.find_open(target_type, target_id)
        if existing is not None:
            return existing

        item = ReviewItem(
            id=f"rv-{uuid.uuid4().hex[:8]}",
            target_type=target_type,
            target_id=target_id,
            title=title or target_id,
            reason=reason,
            proposed_by=proposed_by,
            state=STATE_OPEN,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            payload_preview=payload_preview,
        )
        self._items.append(item)
        return item

    def decide(
        self,
        item_id: str,
        decision: str,
        decided_by: str,
    ) -> ReviewItem:
        """Record an approve/reject decision on an open item.

        Args:
            item_id: The review item id (``rv-xxxx``).
            decision: ``approved`` or ``rejected``.
            decided_by: The reviewer's user id.

        Returns:
            The updated ReviewItem.

        Raises:
            KeyError: if the item id is unknown.
            ValueError: if the decision is invalid or the item isn't open.
        """
        if decision not in (STATE_APPROVED, STATE_REJECTED):
            raise ValueError(
                f"decision must be 'approved' or 'rejected', got {decision!r}"
            )
        for item in self._items:
            if item.id == item_id:
                if item.state != STATE_OPEN:
                    raise ValueError(
                        f"Item {item_id} is already {item.state}; "
                        f"cannot decide again."
                    )
                item.state = decision
                item.decision = decision
                item.decided_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                item.decided_by = decided_by
                return item
        raise KeyError(f"Review item {item_id!r} not found.")

    # ---- queries ----

    def find_open(self, target_type: str, target_id: str) -> Optional[ReviewItem]:
        """Return the open review item for a target, if any."""
        for item in self._items:
            if (
                item.target_type == target_type
                and item.target_id == target_id
                and item.state == STATE_OPEN
            ):
                return item
        return None

    def has_open(self, target_type: str, target_id: str) -> bool:
        return self.find_open(target_type, target_id) is not None

    def list_open(
        self, target_type: Optional[str] = None
    ) -> List[ReviewItem]:
        """List open (pending) items, optionally filtered by target type."""
        return [
            i for i in self._items
            if i.state == STATE_OPEN
            and (target_type is None or i.target_type == target_type)
        ]

    def list_decided(
        self, target_type: Optional[str] = None
    ) -> List[ReviewItem]:
        """List already-decided items (for audit)."""
        return [
            i for i in self._items
            if i.state != STATE_OPEN
            and (target_type is None or i.target_type == target_type)
        ]

    def get(self, item_id: str) -> Optional[ReviewItem]:
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    @property
    def open_count(self) -> int:
        return sum(1 for i in self._items if i.state == STATE_OPEN)
