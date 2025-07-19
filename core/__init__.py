# daemon_7_11_25_refactor/core/__init__.py
"""
Core module for the Daemon AI Assistant
"""

from .orchestrator import DaemonOrchestrator
from .response_generator import ResponseGenerator

__all__ = [
    'DaemonOrchestrator',
    'ResponseGenerator',
]

# Optional imports that may not be ready yet
try:
    from .prompt_builder import UnifiedHierarchicalPromptBuilder
    __all__.append('UnifiedHierarchicalPromptBuilder')
except ImportError:
    pass

try:
    from .query_processor import QueryProcessor
    __all__.append('QueryProcessor')
except ImportError:
    pass
