"""Markdown document parser.

Splits a Markdown document into sections on ATX headings (#, ##, ...)
and records each heading's title and ordinal index in ``Segment.metadata``.
The (index, heading) pair forms a stable section key for change detection
in the source registry: editing body text under a heading keeps the key
intact, while reordering headings changes the index and is correctly
reported as a structural change.

Content that appears before the first heading (preamble) is emitted as a
single section with heading ``""``.
"""

import re
from .base import BaseParser, Segment
from typing import List

# ATX headings: 1-6 '#' followed by text. Captures level and title.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+\s*)?$", re.MULTILINE)


class MarkdownParser(BaseParser):
    def parse(self, file_path: str) -> List[Segment]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        segments: List[Segment] = []
        # Find all heading positions to slice the document.
        matches = list(_HEADING_RE.finditer(content))

        if not matches:
            # No headings - treat whole file as one section.
            if content.strip():
                segments.append(
                    Segment(
                        content=content.strip(),
                        metadata={"heading": "", "index": 0, "level": 0},
                        segment_type="section",
                    )
                )
            return segments

        # Preamble before the first heading (if any).
        if matches[0].start() > 0:
            preamble = content[: matches[0].start()].strip()
            if preamble:
                segments.append(
                    Segment(
                        content=preamble,
                        metadata={"heading": "", "index": 0, "level": 0},
                        segment_type="preamble",
                    )
                )

        # Each heading opens a section running until the next heading.
        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading = m.group(2).strip()
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            if not body:
                continue
            # Keep the original markdown (with leading '#') in content so
            # downstream summarizers see the heading context. Metadata
            # carries the normalized heading + index for stable keys.
            segments.append(
                Segment(
                    content=body,
                    metadata={
                        "heading": heading,
                        "index": i + 1,  # 1-based, skips preamble's 0
                        "level": level,
                    },
                    segment_type=f"h{level}",
                )
            )
        return segments
