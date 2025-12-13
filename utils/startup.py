"""
# utils/startup.py

Module Contract
- Purpose: Staged import system with progress feedback. Updates splash screen during heavy
  imports (torch, sentence_transformers) to show loading progress.
- Inputs:
  - preload flag: Whether to preload optional components
- Outputs:
  - Dictionary of loaded components (model_manager, embedder, etc.)
  - List of (component, error) tuples for any failures
- Key pieces:
  - StartupProgress: Class tracking import progress, updating splash/console
  - run_startup(): Main entry point, imports components in order with timing
  - IMPORT_STAGES: Ordered list of (name, import_func) tuples
- Dependencies:
  - utils.bootstrap (update_splash, close_splash, IS_FROZEN)
- Side effects:
  - Updates PyInstaller splash screen text (frozen mode)
  - Prints progress to console (dev mode)
  - Records timing data for each import stage
- Threading/Async: None (synchronous imports)

Based on audit results (2025-12-12):
- Total startup time: ~12-17s
- Heaviest imports: sentence_transformers (4.1s), torch (1.8s)
- Model loading adds ~1.7s additional

Usage:
    from utils.startup import run_startup, StartupProgress
    components, errors = run_startup()
"""

import sys
import time
from typing import Callable, Optional, Dict, Any, List, Tuple

# Import bootstrap first (handles freeze_support)
from utils.bootstrap import (
    update_splash,
    close_splash,
    IS_FROZEN,
    get_resource_path,
)


class StartupProgress:
    """
    Tracks and displays startup progress.

    In frozen mode: Updates native PyInstaller splash screen
    In dev mode: Prints to console
    """

    def __init__(self, total_steps: int = 10):
        self.steps_completed = 0
        self.total_steps = total_steps
        self.start_time = time.time()
        self.errors: List[Tuple[str, str]] = []
        self.timings: Dict[str, float] = {}

    def update(self, message: str, step: Optional[int] = None) -> None:
        """Update progress display."""
        if step is not None:
            self.steps_completed = step
        else:
            self.steps_completed += 1

        progress = f"[{self.steps_completed}/{self.total_steps}] {message}"

        if IS_FROZEN:
            update_splash(progress)
        else:
            print(f"[Startup] {progress}")

    def error(self, component: str, error: Exception) -> None:
        """Record a non-fatal error."""
        error_msg = str(error)
        self.errors.append((component, error_msg))
        if not IS_FROZEN:
            print(f"[Startup] Warning: {component} failed: {error_msg}")

    def start_timing(self, component: str) -> float:
        """Start timing a component load."""
        return time.time()

    def end_timing(self, component: str, start: float) -> float:
        """End timing and record."""
        elapsed = time.time() - start
        self.timings[component] = elapsed
        return elapsed

    def complete(self) -> List[Tuple[str, str]]:
        """Mark startup as complete."""
        elapsed = time.time() - self.start_time
        if not IS_FROZEN:
            print(f"[Startup] Complete in {elapsed:.1f}s")
            if self.timings:
                print("[Startup] Component timings:")
                for comp, t in sorted(self.timings.items(), key=lambda x: -x[1]):
                    print(f"  {comp}: {t:.2f}s")
        close_splash()
        return self.errors

    def get_elapsed(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time


def load_with_fallback(
    import_func: Callable,
    component_name: str,
    progress: StartupProgress,
    fallback_value: Any = None,
    required: bool = False
) -> Any:
    """
    Load a component with error handling and fallback.

    Args:
        import_func: Function that performs the import/initialization
        component_name: Human-readable name for progress display
        progress: StartupProgress instance
        fallback_value: Value to return if import fails
        required: If True, re-raise exception on failure

    Returns:
        Result of import_func, or fallback_value on failure
    """
    progress.update(f"Loading {component_name}...")
    start = progress.start_timing(component_name)

    try:
        result = import_func()
        progress.end_timing(component_name, start)
        return result
    except Exception as e:
        progress.end_timing(component_name, start)
        progress.error(component_name, e)
        if required:
            raise
        return fallback_value


def staged_import(progress: StartupProgress) -> Dict[str, Any]:
    """
    Perform staged imports with progress updates.

    Returns dict of imported components, with None for failed optional imports.

    Import order is optimized based on audit:
    1. Core Python/config (fast)
    2. Torch (slow, ~1.8s)
    3. Sentence transformers (slowest, ~4.1s)
    4. spaCy (moderate, ~0.5s)
    5. ChromaDB (moderate, ~0.6s)
    6. Gradio (moderate, ~1.4s)
    7. Application modules
    """
    components: Dict[str, Any] = {}

    # Stage 1: Core Python environment (fast, ~0.1s)
    progress.update("Initializing Python environment...", 1)

    # Stage 2: Configuration (fast, ~0.1s)
    def load_config():
        from config.app_config import config
        return config

    components['config'] = load_with_fallback(
        load_config, "configuration", progress, required=True
    )

    # Stage 3: Torch (slow, ~1.8s)
    def load_torch():
        import torch
        return torch

    components['torch'] = load_with_fallback(
        load_torch, "PyTorch", progress, required=True
    )

    # Stage 4: Sentence Transformers (slowest, ~4.1s)
    # This includes transformers import
    def load_sentence_transformers():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer

    components['sentence_transformers'] = load_with_fallback(
        load_sentence_transformers, "sentence transformers", progress, required=True
    )

    # Stage 5: spaCy (moderate, ~0.5s)
    def load_spacy():
        import spacy
        return spacy

    components['spacy'] = load_with_fallback(
        load_spacy, "spaCy NLP", progress, required=False
    )

    # Stage 6: ChromaDB (moderate, ~0.6s)
    def load_chromadb():
        import chromadb
        return chromadb

    components['chromadb'] = load_with_fallback(
        load_chromadb, "vector database", progress, required=True
    )

    # Stage 7: Gradio (moderate, ~1.4s)
    def load_gradio():
        import gradio as gr
        return gr

    components['gradio'] = load_with_fallback(
        load_gradio, "GUI framework", progress, required=True
    )

    # Stage 8: Memory Coordinator (moderate, ~1.5s due to cross-encoder)
    def load_memory():
        from memory.memory_coordinator import MemoryCoordinator
        return MemoryCoordinator

    components['memory_coordinator'] = load_with_fallback(
        load_memory, "memory system", progress, required=True
    )

    # Stage 9: Orchestrator (fast after dependencies loaded)
    def load_orchestrator():
        from core.orchestrator import DaemonOrchestrator
        return DaemonOrchestrator

    components['orchestrator'] = load_with_fallback(
        load_orchestrator, "orchestrator", progress, required=True
    )

    # Stage 10: Final initialization
    progress.update("Finalizing initialization...", 10)

    return components


def preload_models(progress: StartupProgress) -> Dict[str, Any]:
    """
    Pre-load ML models to warm up caches.

    This is optional but improves first-query response time.
    """
    models: Dict[str, Any] = {}

    # Load spaCy model
    def load_spacy_model():
        import spacy
        return spacy.load('en_core_web_sm')

    models['spacy_nlp'] = load_with_fallback(
        load_spacy_model, "spaCy model", progress, required=False
    )

    # Note: Sentence transformer model is loaded on-demand by the embedder
    # Pre-loading it here would add ~1.3s but improve first query

    return models


def run_startup(preload: bool = False) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """
    Complete startup sequence with visual feedback.

    Args:
        preload: If True, pre-load ML models (adds ~1.5s but improves first query)

    Returns:
        tuple: (components dict, list of (component, error) tuples)
    """
    total_steps = 10 if not preload else 12
    progress = StartupProgress(total_steps=total_steps)

    try:
        # Import heavy modules
        components = staged_import(progress)

        # Optionally pre-load models
        if preload:
            models = preload_models(progress)
            components.update(models)

        errors = progress.complete()
        return components, errors

    except Exception as e:
        progress.complete()
        raise


def check_requirements() -> List[str]:
    """
    Check if all required packages are available.

    Returns list of missing packages.
    """
    required = [
        'torch',
        'transformers',
        'sentence_transformers',
        'chromadb',
        'gradio',
        'spacy',
        'tiktoken',
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)

    return missing


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("STARTUP MODULE TEST")
    print("=" * 60)

    print("\nChecking requirements...")
    missing = check_requirements()
    if missing:
        print(f"  Missing packages: {missing}")
    else:
        print("  All required packages available")

    print("\nRunning startup sequence...")
    print("-" * 60)

    try:
        components, errors = run_startup(preload=False)

        print("-" * 60)
        print(f"\nComponents loaded: {len(components)}")
        for name, comp in components.items():
            status = "OK" if comp is not None else "FAILED"
            print(f"  {name}: {status}")

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for comp, err in errors:
                print(f"  {comp}: {err}")
        else:
            print("\nNo errors during startup")

    except Exception as e:
        print(f"\nStartup failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
