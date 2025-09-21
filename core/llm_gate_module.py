#core/llm_gate_module.py
"""Compatibility shim re-exporting the gated prompt builder."""

from processing.gate_system import GatedPromptBuilder

__all__ = ["GatedPromptBuilder"]

