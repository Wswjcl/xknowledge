"""Word document parser."""

from .base import BaseParser, Segment
from typing import List


class WordParser(BaseParser):
    def parse(self, file_path: str) -> List[Segment]:
        try:
            from docx import Document
            doc = Document(file_path)
            segments = []
            for para in doc.paragraphs:
                if para.text.strip():
                    segments.append(Segment(content=para.text.strip(), segment_type="paragraph"))
            return segments
        except ImportError:
            raise ImportError("python-docx is required. Install with: pip install qawiki[parsers]")
