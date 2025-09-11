# core/__init__.py
"""
Core module initialization - controls import order to prevent circular dependencies

Import order:
1. Dependencies and configs
2. Base classes and interfaces
3. Implementations
4. High-level orchestrators
"""

# These get imported first (no dependencies on other core modules)
from .dependencies import deps

# Import order matters - least dependent to most dependent
__all__ = [
    'deps',
    'ResponseGenerator',
    'DaemonOrchestrator',
    'UnifiedPromptBuilder'
]

# Lazy imports to prevent circular dependencies
def get_orchestrator_class():
    from .orchestrator import DaemonOrchestrator
    return DaemonOrchestrator

def get_prompt_builder_class():
    from .prompt_builder_v2 import UnifiedPromptBuilder
    return UnifiedPromptBuilder

def get_response_generator_class():
    from .response_generator import ResponseGenerator
    return ResponseGenerator
