"""Utility functions - image encoding, insight library I/O.
Adapted from XSkill eval/exskill/experience_utils.py
Original work by Jiang et al. (ICML 2026, MIT License)
"""

import os
import json
import base64
import io
from typing import Dict
from PIL import Image
from ..prompts.query import INSIGHT_INJECTION_HEADER


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI.
    
    Args:
        image: PIL Image to convert
        
    Returns:
        Base64-encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --------- Experience Library I/O ---------

def load_insight_library(path: str) -> Dict[str, str]:
    """Load existing insights from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary mapping insight IDs to insight text
    """
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, dict) and "insights" in data:
            return data["insights"]
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


load_existing = load_insight_library


def save_insight_library(path: str, experiences: Dict[str, str]):
    """Save insights to a JSON file.
    
    Args:
        path: Path to save the JSON file
        experiences: Dictionary mapping insight IDs to insight text
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"insights": experiences}, f, ensure_ascii=False, indent=2)


save_library = save_insight_library


def load_insights(path: str) -> Dict[str, str]:
    """Load insights from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary mapping insight IDs to insight text
    """
    if path and os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(data, dict) and "insights" in data:
                return data["insights"]
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


load_experiences = load_insights


def format_insights_for_prompt(experiences: Dict[str, str], max_items: int = 32) -> str:
    """Format insights for injection into prompts.
    
    Args:
        experiences: Dictionary mapping insight IDs to insight text
        max_items: Maximum number of insights to include
        
    Returns:
        Formatted string for prompt injection
    """
    if not experiences:
        return ""
    items = list(experiences.items())[:max_items]
    bullets = "\n".join([f"- [{k}] {v}" for k, v in items])
    return INSIGHT_INJECTION_HEADER.format(bullets=bullets)


format_for_prompt = format_insights_for_prompt

