"""Role-based access control for knowledge governance.

Four roles, least-privileged to most:

    viewer < editor < reviewer < admin

Each role can perform a set of Actions. An Action is either:
  * a **transition action** (matches ``governance.states`` action names:
    submit / approve / reject / deprecate / reopen / rollback / withdraw),
  * or a **doc action** (delete / manage_permissions).

Permission is decided by ``can(actor_role, action)``. Callers should call
this *before* attempting a state transition or destructive op.

Identity model: a minimal ``Actor`` carries a user id + role. Roles are
resolved per-document from the document manifest's ``permissions`` map,
falling back to a global default role (configurable). This keeps the
model simple (no auth system) while supporting the "some people can
update the shared KB directly, others can't" requirement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional, Set


class Role(str, Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    ADMIN = "admin"


# Role rank for "at least" comparisons.
_RANK = {
    Role.VIEWER: 0,
    Role.EDITOR: 1,
    Role.REVIEWER: 2,
    Role.ADMIN: 3,
}


class Action(str, Enum):
    # Read
    VIEW_PUBLISHED = "view_published"
    VIEW_PENDING = "view_pending"
    # Transitions (names must match governance.states._ACTION_NAME values)
    SUBMIT = "submit"            # draft -> pending
    APPROVE = "approve"          # pending -> published
    REJECT = "reject"            # pending -> draft
    WITHDRAW = "withdraw"        # pending -> deprecated
    DEPRECATE = "deprecate"      # published -> deprecated
    REOPEN = "reopen"            # published -> pending
    ROLLBACK = "rollback"        # deprecated -> published
    # Destructive / structural
    DELETE = "delete"
    MANAGE_PERMISSIONS = "manage_permissions"
    # Bypass review: publish content immediately without pending step.
    BYPASS_REVIEW = "bypass_review"


# The permission matrix: role -> actions it may perform.
# Inherited cumulatively (admin can do everything reviewer can, etc.)
_MATRIX: Dict[Role, FrozenSet[Action]] = {
    Role.VIEWER: frozenset({
        Action.VIEW_PUBLISHED,
    }),
    Role.EDITOR: frozenset({
        Action.VIEW_PUBLISHED,
        Action.VIEW_PENDING,
        Action.SUBMIT,
        Action.REOPEN,           # reopen own published item for editing
    }),
    Role.REVIEWER: frozenset({
        Action.VIEW_PUBLISHED,
        Action.VIEW_PENDING,
        Action.SUBMIT,
        Action.APPROVE,
        Action.REJECT,
        Action.WITHDRAW,
        Action.DEPRECATE,
        Action.REOPEN,
        Action.BYPASS_REVIEW,    # reviewer may publish directly
    }),
    Role.ADMIN: frozenset({
        Action.VIEW_PUBLISHED,
        Action.VIEW_PENDING,
        Action.SUBMIT,
        Action.APPROVE,
        Action.REJECT,
        Action.WITHDRAW,
        Action.DEPRECATE,
        Action.REOPEN,
        Action.ROLLBACK,
        Action.DELETE,
        Action.MANAGE_PERMISSIONS,
        Action.BYPASS_REVIEW,
    }),
}


# Map state-transition action names (from states.py) to the Action that
# authorizes them. Keep in sync with states._ACTION_NAME.
_TRANSITION_ACTION: Dict[str, Action] = {
    "submit": Action.SUBMIT,
    "approve": Action.APPROVE,
    "reject": Action.REJECT,
    "withdraw": Action.WITHDRAW,
    "deprecate": Action.DEPRECATE,
    "reopen": Action.REOPEN,
    "rollback": Action.ROLLBACK,
}


class PermissionDenied(Exception):
    """Raised when an actor lacks permission for an action."""


@dataclass
class Actor:
    """A user acting on the knowledge base.

    Attributes:
        user_id: Stable identifier (e.g. username / employee id).
        role: Effective role for the operation. Callers should resolve
            this per-document via ``resolve_role`` before constructing.
        doc_roles: Optional per-document role overrides, for ad-hoc use
            without a manifest (mostly for tests).
    """

    user_id: str
    role: Role = Role.VIEWER
    doc_roles: Dict[str, Role] = field(default_factory=dict)

    def role_for(self, doc_id: Optional[str]) -> Role:
        """Effective role for a given document (override wins)."""
        if doc_id and doc_id in self.doc_roles:
            return self.doc_roles[doc_id]
        return self.role


def coerce_role(value) -> Role:
    if isinstance(value, Role):
        return value
    if isinstance(value, str):
        try:
            return Role(value)
        except ValueError:
            raise ValueError(
                f"Unknown role {value!r}; expected one of "
                f"{[r.value for r in Role]}"
            )
    raise TypeError(f"Cannot coerce {type(value).__name__} into Role")


def can(role: Role, action: Action) -> bool:
    """True iff ``role`` is permitted to perform ``action``."""
    role = coerce_role(role)
    action = Action(action) if not isinstance(action, Action) else action
    return action in _MATRIX.get(role, frozenset())


def require(role: Role, action: Action):
    """Raise PermissionDenied if ``role`` cannot perform ``action``."""
    if not can(role, action):
        raise PermissionDenied(
            f"Role {coerce_role(role).value!r} cannot perform "
            f"{Action(action).value!r}."
        )


def can_transition(role: Role, transition_action: str) -> bool:
    """Authorize a state-transition action by its name (e.g. 'approve')."""
    action = _TRANSITION_ACTION.get(transition_action)
    if action is None:
        return False
    return can(role, action)


def require_transition(role: Role, transition_action: str):
    """Raise PermissionDenied if role cannot perform the transition."""
    action = _TRANSITION_ACTION.get(transition_action)
    if action is None:
        raise PermissionDenied(
            f"Unknown transition action {transition_action!r}."
        )
    require(role, action)


def resolve_role(
    actor: Actor,
    doc_id: Optional[str],
    manifest_perms: Optional[Dict[str, list]] = None,
    default_role: Role = Role.VIEWER,
) -> Role:
    """Resolve an actor's effective role for a document.

    Precedence (highest first):
      1. Actor's per-document override (``actor.doc_roles[doc_id]``)
      2. Document manifest's role lists (``manifest_perms[role_value]``
         contains user_ids), with admin > reviewer > editor > viewer.
      3. The actor's global ``actor.role``.
      4. ``default_role``.

    ``manifest_perms`` follows the shape used by reference/insight
    manifests: ``{"editors": [...], "reviewers": [...], "admins": [...]}``.
    """
    if doc_id and doc_id in actor.doc_roles:
        return actor.doc_roles[doc_id]

    if manifest_perms and actor.user_id:
        # Find the highest role whose list contains this user.
        for role in (Role.ADMIN, Role.REVIEWER, Role.EDITOR, Role.VIEWER):
            members = manifest_perms.get(role.value + "s") or \
                manifest_perms.get(role.value) or []
            if actor.user_id in members:
                return role

    if actor.role is not None:
        return actor.role
    return default_role


def effective_actions(role: Role) -> FrozenSet[Action]:
    """Return the full action set a role may perform (for diagnostics)."""
    return _MATRIX.get(coerce_role(role), frozenset())
