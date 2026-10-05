"""Ghana Assistant: a small model (Flan-T5-small) with grounded skills for Ghana.
- navigation: landmark-based spoken directions from a road graph (Accra, Kumasi)
- knowledge: short answers grounded in retrieved Ghanaian news and research sentences
"""
from .assistant import GhanaAssistant

__all__ = ["GhanaAssistant"]
__version__ = "0.1.0"
