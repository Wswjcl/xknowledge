"""Knowledge-item lifecycle states.

Every knowledge artifact in the KB (an insight, a reference doc version,
a framework revision) carries one of these states. The state machine
enforces the team-governance rule that **automatically distilled
content is published only after review**.

States
------
    draft ──submit──▶ pending ──approve──▶ published ──new version──▶ deprecated
                          │
                          └──reject──▶ draft  (or removed)

Rules
-----
* ``draft``     : being authored; not visible to viewers.
* ``pending``   : submitted / auto-distilled; awaiting review. Visible to
                  editor+ but NOT served by default retrieval.
* ``published`` : live; the only state default retrieval returns.
* ``deprecated``: superseded by a newer version or withdrawn. Kept for
                  audit / rollback, not served.

Transitions are policy-checked against a role via ``governance.permissions``.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, FrozenSet, Optional, Tuple


class State(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


# The set of states that default retrieval should return to end-users.
PUBLISHED_LIKE: FrozenSet[State] = frozenset({State.PUBLISHED})

# States visible to contributors (editor+). Viewers only see published.
CONTRIBUTOR_VISIBLE: FrozenSet[State] = frozenset(
    {State.DRAFT, State.PENDING, State.PUBLISHED, State.DEPRECATED}
)


class IllegalTransition(Exception):
    """Raised when a state transition is not allowed."""


# Allowed (from_state -> {to_state}). Anything not listed is rejected.
_ALLOWED: Dict[State, FrozenSet[State]] = {
    State.DRAFT: frozenset({State.PENDING}),
    State.PENDING: frozenset({State.PUBLISHED, State.DRAFT, State.DEPRECATED}),
    State.PUBLISHED: frozenset({State.DEPRECATED, State.PENDING}),
    # deprecated is terminal-ish; it can be re-published (rollback) only.
    State.DEPRECATED: frozenset({State.PUBLISHED}),
}

# Human-readable action name for each (from, to) pair, for audit logs.
_ACTION_NAME: Dict[Tuple[State, State], str] = {
    (State.DRAFT, State.PENDING): "submit",
    (State.PENDING, State.PUBLISHED): "approve",
    (State.PENDING, State.DRAFT): "reject",
    (State.PENDING, State.DEPRECATED): "withdraw",
    (State.PUBLISHED, State.DEPRECATED): "deprecate",
    (State.PUBLISHED, State.PENDING): "reopen",
    (State.DEPRECATED, State.PUBLISHED): "rollback",
}


@dataclass(frozen=True)
class Transition:
    """A validated state transition with audit metadata."""

    from_state: State
    to_state: State
    action: str

    def __post_init__(self):
        if self.to_state not in _ALLOWED.get(self.from_state, frozenset()):
            raise IllegalTransition(
                f"Transition {self.from_state.value} -> {self.to_state.value} "
                f"is not allowed."
            )


def transition(from_state: State, to_state: State) -> Transition:
    """Build a validated transition or raise IllegalTransition.

    Args:
        from_state: Current state of the item.
        to_state: Desired next state.

    Returns:
        A Transition object carrying the canonical action name.

    Raises:
        IllegalTransition: If the (from, to) pair is not in the allowed set.
    """
    if not isinstance(from_state, State):
        from_state = State(from_state)
    if not isinstance(to_state, State):
        to_state = State(to_state)
    action = _ACTION_NAME.get((from_state, to_state), "transition")
    return Transition(from_state, to_state, action)


def can_transition(from_state: State, to_state: State) -> bool:
    """True iff the transition is allowed (no exception)."""
    try:
        transition(from_state, to_state)
        return True
    except IllegalTransition:
        return False


def action_for(from_state: State, to_state: State) -> Optional[str]:
    """Return the canonical action name for a transition, or None."""
    try:
        return transition(from_state, to_state).action
    except IllegalTransition:
        return None


def coerce(value) -> State:
    """Coerce a string/State into a State, raising ValueError on bad input."""
    if isinstance(value, State):
        return value
    if isinstance(value, str):
        try:
            return State(value)
        except ValueError:
            raise ValueError(
                f"Unknown state {value!r}; expected one of "
                f"{[s.value for s in State]}"
            )
    raise TypeError(f"Cannot coerce {type(value).__name__} into State")


def is_served_by_default(state) -> bool:
    """Whether default (viewer) retrieval serves items in this state."""
    return coerce(state) in PUBLISHED_LIKE

