"""Markdown document parser."""

from .base import BaseParser, Segment
from typing import List


class MarkdownParser(BaseParser):
    def parse(self, file_path: str) -> List[Segment]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Simple section-based splitting
        sections = content.split("\n## ")
        segments = []
        for sec in sections:
            if sec.strip():
                segments.append(Segment(content=sec.strip(), segment_type="section"))
        return segments
