"""Word document parser.

Groups consecutive paragraphs under their nearest preceding Word heading
style (Heading 1-9). Each group becomes one section whose
``Segment.metadata`` records the heading title and an ordinal index,
mirroring the Markdown parser so both feed the same section-key scheme.

When no heading styles are present, each non-empty paragraph is emitted
as its own section keyed by its index (degraded but still functional).
"""

from .base import BaseParser, Segment
from typing import List

_HEADING_STYLE_PREFIX = "Heading"


def _is_heading(style_name: str) -> bool:
    return bool(style_name) and (
        style_name.startswith(_HEADING_STYLE_PREFIX)
        or style_name == "Title"
    )


class WordParser(BaseParser):
    def parse(self, file_path: str) -> List[Segment]:
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required. Install with: pip install qawiki[parsers]"
            )

        doc = Document(file_path)

        # First pass: collect (style, text) for non-empty paragraphs,
        # preserving document order.
        items = []
        for para in doc.paragraphs:
            text = (para.text or "").strip()
            if not text:
                continue
            style_name = para.style.name if para.style is not None else ""
            items.append((style_name, text))

        segments: List[Segment] = []

        # No headings at all -> one section per paragraph, indexed.
        if not any(_is_heading(style) for style, _ in items):
            for i, (_, text) in enumerate(items):
                segments.append(
                    Segment(
                        content=text,
                        metadata={"heading": "", "index": i + 1, "level": 0},
                        segment_type="paragraph",
                    )
                )
            return segments

        # Group paragraphs under their nearest preceding heading.
        current_heading = ""
        current_level = 0
        buffer = []
        index = 0

        def flush(idx):
            if not buffer:
                return
            segments.append(
                Segment(
                    content="\n".join(buffer).strip(),
                    metadata={
                        "heading": current_heading,
                        "index": idx,
                        "level": current_level,
                    },
                    segment_type=f"h{current_level}" if current_level else "section",
                )
            )

        for style_name, text in items:
            if _is_heading(style_name):
                # Close out the previous section.
                if buffer:
                    index += 1
                    flush(index)
                    buffer = []
                current_heading = text
                # Parse "Heading 2" -> 2, "Title" -> 1.
                level = 1
                if style_name.startswith(_HEADING_STYLE_PREFIX):
                    tail = style_name[len(_HEADING_STYLE_PREFIX):].strip()
                    if tail.isdigit():
                        level = int(tail)
                current_level = level
                # A heading with no body is still a meaningful boundary;
                # include it as the start of the new section's content.
                buffer = [text]
            else:
                buffer.append(text)

        if buffer:
            index += 1
            flush(index)

        return segments
