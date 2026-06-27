"""Governance layer: states, permissions, and the review queue.

This package is deliberately **cross-cutting**: it wraps every write path
into the knowledge base. The core distillation/retrieval modules remain
unaware of governance; ``knowledge.py`` (and the CLI) enforce states and
permissions around them.

Quick start:
    from xknowledge.governance import (
        State, transition, Action, Role, Actor,
        can, require, resolve_role, ReviewQueue,
    )
"""

from .states import (
    State,
    Transition,
    IllegalTransition,
    PUBLISHED_LIKE,
    CONTRIBUTOR_VISIBLE,
    transition,
    can_transition,
    action_for,
    coerce as coerce_state,
    is_served_by_default,
)
from .permissions import (
    Role,
    Action,
    Actor,
    PermissionDenied,
    can,
    require,
    can_transition as role_can_transition,
    require_transition,
    resolve_role,
    effective_actions,
    coerce_role,
)
from .review import (
    ReviewQueue,
    ReviewItem,
    TARGET_INSIGHT,
    TARGET_REFERENCE,
    TARGET_FRAMEWORK,
)

__all__ = [
    # states
    "State", "Transition", "IllegalTransition",
    "PUBLISHED_LIKE", "CONTRIBUTOR_VISIBLE",
    "transition", "can_transition", "action_for",
    # permissions
    "Role", "Action", "Actor", "PermissionDenied",
    "can", "require", "role_can_transition", "require_transition",
    "resolve_role", "effective_actions", "coerce_role",
    # review
    "ReviewQueue", "ReviewItem",
    "TARGET_INSIGHT", "TARGET_REFERENCE", "TARGET_FRAMEWORK",
]
