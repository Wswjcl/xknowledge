"""Base parser interface."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Segment:
    content: str
    metadata: dict = field(default_factory=dict)
    segment_type: str = "paragraph"


class BaseParser:
    def parse(self, file_path: str) -> List[Segment]:
        raise NotImplementedError
