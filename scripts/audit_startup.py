#!/usr/bin/env python3
"""
Audit script to document startup chain and import timing.

This script measures the import time of heavy dependencies to establish
a baseline for executable startup optimization.

Run: python scripts/audit_startup.py
"""
import time
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def timed_import(module_name):
    """Time how long an import takes."""
    # Check if already imported
    if module_name in sys.modules:
        print(f"  {module_name}: already imported (skipped)")
        return 0.0

    start = time.time()
    try:
        __import__(module_name)
        elapsed = time.time() - start
        print(f"  {module_name}: {elapsed:.2f}s")
        return elapsed
    except ImportError as e:
        print(f"  {module_name}: MISSING - {e}")
        return 0.0
    except Exception as e:
        print(f"  {module_name}: ERROR - {e}")
        return 0.0

def audit_heavy_imports():
    """Audit heavy module import times."""
    print("=" * 60)
    print("IMPORT TIMING AUDIT")
    print("=" * 60)

    # Core heavy imports (ML/AI)
    print("\n[1] Core ML Libraries:")
    ml_modules = [
        'torch',
        'transformers',
        'sentence_transformers',
        'chromadb',
        'faiss',
    ]
    ml_total = sum(timed_import(m) for m in ml_modules)
    print(f"  Subtotal: {ml_total:.2f}s")

    # NLP
    print("\n[2] NLP Libraries:")
    nlp_modules = [
        'spacy',
        'tiktoken',
        'nltk',
    ]
    nlp_total = sum(timed_import(m) for m in nlp_modules)
    print(f"  Subtotal: {nlp_total:.2f}s")

    # Web/GUI
    print("\n[3] Web/GUI Libraries:")
    web_modules = [
        'gradio',
        'fastapi',
        'uvicorn',
    ]
    web_total = sum(timed_import(m) for m in web_modules)
    print(f"  Subtotal: {web_total:.2f}s")

    # Utilities
    print("\n[4] Utility Libraries:")
    util_modules = [
        'yaml',
        'aiohttp',
        'requests',
        'pydantic',
    ]
    util_total = sum(timed_import(m) for m in util_modules)
    print(f"  Subtotal: {util_total:.2f}s")

    total = ml_total + nlp_total + web_total + util_total
    print("\n" + "=" * 60)
    print(f"TOTAL HEAVY IMPORT TIME: {total:.2f}s")
    print("=" * 60)
    print("\nThis is the MINIMUM startup delay before any GUI appears.")

    return total

def audit_project_imports():
    """Audit project-specific module import times."""
    print("\n" + "=" * 60)
    print("PROJECT MODULE IMPORT TIMING")
    print("=" * 60)

    # Project modules (after dependencies are loaded)
    print("\n[5] Project Modules:")
    project_modules = [
        'config.app_config',
        'utils.logging_utils',
        'utils.time_manager',
        'models.model_manager',
        'memory.corpus_manager',
        'memory.memory_coordinator',
        'core.orchestrator',
        'core.response_generator',
        'gui.launch',
    ]

    project_total = 0.0
    for mod in project_modules:
        project_total += timed_import(mod)

    print(f"  Subtotal: {project_total:.2f}s")
    return project_total

def audit_model_loading():
    """Audit model loading times (separate from imports)."""
    print("\n" + "=" * 60)
    print("MODEL LOADING TIMING")
    print("=" * 60)

    # spaCy model
    print("\n[6] spaCy Model (en_core_web_sm):")
    try:
        import spacy
        start = time.time()
        nlp = spacy.load('en_core_web_sm')
        elapsed = time.time() - start
        print(f"  spacy.load('en_core_web_sm'): {elapsed:.2f}s")
    except Exception as e:
        print(f"  spacy.load failed: {e}")
        elapsed = 0.0
    spacy_time = elapsed

    # Sentence transformers model
    print("\n[7] Sentence Transformers Model:")
    try:
        from sentence_transformers import SentenceTransformer
        start = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        elapsed = time.time() - start
        print(f"  SentenceTransformer('all-MiniLM-L6-v2'): {elapsed:.2f}s")
    except Exception as e:
        print(f"  SentenceTransformer load failed: {e}")
        elapsed = 0.0
    st_time = elapsed

    print(f"\n  Model loading subtotal: {spacy_time + st_time:.2f}s")
    return spacy_time + st_time

def audit_entry_points():
    """Document entry points in main.py."""
    print("\n" + "=" * 60)
    print("ENTRY POINT ANALYSIS")
    print("=" * 60)

    main_py_path = os.path.join(project_root, 'main.py')

    if not os.path.exists(main_py_path):
        print("  ERROR: main.py not found")
        return

    with open(main_py_path, 'r') as f:
        content = f.read()

    # Find modes
    modes = []
    if 'mode == "gui"' in content or 'mode == "gui"' in content:
        modes.append('gui')
    if 'mode == "cli"' in content:
        modes.append('cli')
    if 'mode == "wizard"' in content:
        modes.append('wizard')
    if 'mode == "test-summaries"' in content:
        modes.append('test-summaries')
    if 'mode == "inspect-summaries"' in content:
        modes.append('inspect-summaries')
    if 'mode == "export-profile"' in content:
        modes.append('export-profile')
    if 'mode == "show-profile"' in content:
        modes.append('show-profile')

    print(f"\n  Entry modes found: {modes}")
    print(f"  Default mode: gui")

    # Check for multiprocessing
    has_freeze_support = 'freeze_support' in content
    print(f"\n  Has multiprocessing.freeze_support(): {has_freeze_support}")

    # Check for signal handlers
    has_signal_handlers = 'signal.SIGTERM' in content or 'signal.SIGINT' in content
    print(f"  Has signal handlers: {has_signal_handlers}")

    # Check for wizard routing
    has_wizard_routing = 'force_wizard' in content
    print(f"  Has wizard routing: {has_wizard_routing}")

def check_data_locations():
    """Check current data file locations."""
    print("\n" + "=" * 60)
    print("DATA LOCATION AUDIT")
    print("=" * 60)

    data_files = [
        ('data/corpus_v4.json', 'Conversation corpus'),
        ('data/chroma_db_v4', 'ChromaDB vector store'),
        ('data/user_profile.json', 'User profile'),
        ('data/last_query_time.json', 'Query timestamp'),
        ('data/last_session_time.json', 'Session timestamp'),
        ('.env', 'Environment variables'),
        ('config/config.yaml', 'YAML configuration'),
    ]

    print("\n  Current data locations:")
    for path, desc in data_files:
        full_path = os.path.join(project_root, path)
        exists = os.path.exists(full_path)
        if os.path.isdir(full_path):
            size = "directory"
        elif exists:
            size = f"{os.path.getsize(full_path) / 1024:.1f} KB"
        else:
            size = "missing"
        print(f"    {path}: {size} ({desc})")

def main():
    """Run complete audit."""
    print("\n" + "=" * 60)
    print("DAEMON STARTUP AUDIT")
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print("=" * 60)

    total_time = 0.0

    # Run audits
    total_time += audit_heavy_imports()
    total_time += audit_project_imports()
    total_time += audit_model_loading()

    audit_entry_points()
    check_data_locations()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Total measured startup time: {total_time:.2f}s")
    print(f"  Expected cold start to GUI: {total_time + 2:.0f}-{total_time + 5:.0f}s")
    print("\n  Key findings:")
    print("    - Heavy imports (torch, transformers) dominate startup")
    print("    - Model loading adds significant time on first run")
    print("    - Splash screen should appear within 1-2s for good UX")
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
