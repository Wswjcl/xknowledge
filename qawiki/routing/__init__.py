"""Source routing: decides how each document enters the knowledge base.

Re-exports the classifier and canonical source-type constants.
"""

from .classifier import (
    DISTILLABLE,
    REFERENCE,
    Classification,
    ClassificationError,
    classify_source,
    origin_for,
)

__all__ = [
    "DISTILLABLE",
    "REFERENCE",
    "Classification",
    "ClassificationError",
    "classify_source",
    "origin_for",
]
