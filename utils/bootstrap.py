#!/usr/bin/env python3
"""
# utils/bootstrap.py

Module Contract
- Purpose: Bootstrap frozen executables. Handles path resolution, data migration, and environment
  setup BEFORE any other Daemon imports. Critical for PyInstaller builds.
- Inputs:
  - sys.frozen attribute (True when running as PyInstaller executable)
  - Platform detection (Windows, macOS, Linux)
- Outputs:
  - Environment variables set: CORPUS_FILE, CHROMA_PATH, USER_PROFILE_PATH, etc.
  - User data directory created at platform-specific location
  - Data migration from old locations if needed
- Key pieces:
  - IS_FROZEN: Boolean detecting PyInstaller frozen mode
  - get_app_dir(): Returns executable directory (frozen) or project root (dev)
  - get_user_data_dir(): Returns ~/.daemon (Linux), %APPDATA%/Daemon (Windows), etc.
  - setup_environment(): Sets all env vars, loads .env, migrates data
  - close_splash(), update_splash(): PyInstaller splash screen control
  - initialize(): Full initialization returning config dict
- Dependencies:
  - Standard library only (sys, os, shutil, json, pathlib)
  - Optional: dotenv for .env loading, pyi_splash for splash screen
- Side effects:
  - Creates ~/.daemon/ directory structure
  - Sets os.environ variables for paths
  - Migrates data from dist/data/ to user data directory
- Threading/Async: None (synchronous module-level initialization)

CRITICAL: Must be imported BEFORE config.app_config to set environment variables.

USAGE in main.py:
    if getattr(sys, 'frozen', False):
        from utils.bootstrap import setup_environment
        setup_environment()  # Sets CORPUS_FILE, etc.
    # Now config imports will read correct paths
    from config.app_config import config
"""

import sys
import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional

# =============================================================================
# FROZEN DETECTION
# =============================================================================

IS_FROZEN = getattr(sys, 'frozen', False)
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')

# =============================================================================
# PATH RESOLUTION
# =============================================================================

def get_app_dir() -> str:
    """
    Get the application directory.

    Frozen: Directory containing the executable
    Development: Directory containing main.py (project root)
    """
    if IS_FROZEN:
        # PyInstaller sets sys.executable to the exe path
        # For one-dir mode, this is in the app folder
        return os.path.dirname(sys.executable)
    else:
        # Development mode: project root (parent of utils/)
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to a bundled resource.

    For frozen one-dir mode: looks in app directory structure
    For development: looks relative to project root

    Args:
        relative_path: Path relative to project root (e.g., 'core/system_prompt.txt')

    Returns:
        Absolute path to the resource
    """
    if IS_FROZEN:
        base_path = get_app_dir()
        # Try _internal first (PyInstaller default for data in one-dir)
        internal_path = os.path.join(base_path, '_internal', relative_path)
        if os.path.exists(internal_path):
            return internal_path
        # Fall back to app root
        return os.path.join(base_path, relative_path)
    else:
        # Development mode
        return os.path.join(get_app_dir(), relative_path)


def get_user_data_dir() -> str:
    """
    Get user-writable data directory for mutable files.

    This is where we store:
    - corpus_v4.json (conversation history)
    - chroma_db_v4/ (vector store)
    - user_profile.json
    - .env (API keys)
    - logs/
    - conversation_logs/

    Platform locations:
    - Windows: %APPDATA%/Daemon
    - macOS: ~/Library/Application Support/Daemon
    - Linux: ~/.daemon

    In development mode, returns ./data/ for backward compatibility.
    """
    if not IS_FROZEN:
        # Development mode: use project's data/ directory
        return os.path.join(get_app_dir(), 'data')

    # Frozen mode: use platform-specific user directory
    if IS_WINDOWS:
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
        return os.path.join(base, 'Daemon')
    elif IS_MACOS:
        return os.path.expanduser('~/Library/Application Support/Daemon')
    else:  # Linux and others
        return os.path.expanduser('~/.daemon')


def get_external_data_dir() -> Optional[str]:
    """
    Get directory for large external data (Wiki, FAISS index).

    This is separate from user data because:
    1. It's optional (feature works without it)
    2. It's huge (100GB+)
    3. User might want it on a different drive

    Checks in order:
    1. DAEMON_EXTERNAL_DATA environment variable
    2. User data dir /external/
    3. None (feature disabled)
    """
    # Check environment override first
    env_path = os.environ.get('DAEMON_EXTERNAL_DATA')
    if env_path and os.path.isdir(env_path):
        return env_path

    # Default location in user data
    default_path = os.path.join(get_user_data_dir(), 'external')
    if os.path.isdir(default_path):
        return default_path

    return None


# =============================================================================
# DIRECTORY INITIALIZATION
# =============================================================================

def ensure_directories() -> str:
    """
    Create all required directories.

    Returns:
        Path to user data directory
    """
    user_dir = get_user_data_dir()

    # Create user data directory structure
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(os.path.join(user_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(user_dir, 'conversation_logs'), exist_ok=True)

    # Initialize empty corpus if missing
    corpus_path = os.path.join(user_dir, 'corpus_v4.json')
    if not os.path.exists(corpus_path):
        with open(corpus_path, 'w') as f:
            f.write('[]')

    return user_dir


# =============================================================================
# DATA MIGRATION
# =============================================================================

def migrate_user_data() -> None:
    """
    Migrate user data from old locations if needed.

    This handles users who:
    1. Used the development version before
    2. Upgraded from an older executable version
    3. Have data in project's ./data/ directory

    Migration is one-way: dev -> user data dir (for frozen mode only)
    """
    if not IS_FROZEN:
        # In development mode, we use ./data/ directly, no migration needed
        return

    user_dir = get_user_data_dir()

    # Old development locations to check for migration
    # Priority order: most likely locations first
    old_locations = [
    # Inside app folder (bundled data only)
    os.path.join(get_app_dir(), 'data'),
    ]

    # Files and directories to migrate
    items_to_migrate = [
        ('corpus_v4.json', False),       # File
        ('user_profile.json', False),    # File
        ('chroma_db_v4', True),          # Directory
        ('last_query_time.json', False), # File
        ('last_session_time.json', False), # File
        # Note: .env is NOT migrated - it's created fresh by wizard
    ]

    migrated_any = False

    for old_dir in old_locations:
        if not os.path.isdir(old_dir):
            continue

        for item_name, is_dir in items_to_migrate:
            old_path = os.path.join(old_dir, item_name)
            new_path = os.path.join(user_dir, item_name)

            # Only migrate if old exists and new doesn't
            if not os.path.exists(old_path):
                continue
            if os.path.exists(new_path):
                continue

            try:
                if is_dir:
                    shutil.copytree(old_path, new_path)
                else:
                    shutil.copy2(old_path, new_path)
                print(f"[Bootstrap] Migrated {item_name} from {old_dir}")
                migrated_any = True
            except Exception as e:
                print(f"[Bootstrap] Warning: Could not migrate {item_name}: {e}")

    if migrated_any:
        print(f"[Bootstrap] Data migration complete. Data now in: {user_dir}")


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment() -> str:
    """
    Configure environment variables for frozen executable.

    CRITICAL: Must be called before importing any Daemon modules that
    read environment variables (especially config.app_config).

    Returns:
        Path to user data directory
    """
    user_dir = ensure_directories()

    # Migrate old data if running frozen
    migrate_user_data()

    # Load .env from user data directory (not bundled resources)
    env_path = os.path.join(user_dir, '.env')
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
        except ImportError:
            # Manual parsing if dotenv not available
            _load_env_manual(env_path)

    # Set default paths for frozen mode
    # These environment variables are read by config/app_config.py
    if IS_FROZEN:
        # User data paths
        os.environ.setdefault('CORPUS_FILE', os.path.join(user_dir, 'corpus_v4.json'))
        os.environ.setdefault('CHROMA_PATH', os.path.join(user_dir, 'chroma_db_v4'))
        os.environ.setdefault('USER_PROFILE_PATH', os.path.join(user_dir, 'user_profile.json'))
        os.environ.setdefault('LOG_DIR', os.path.join(user_dir, 'logs'))
        os.environ.setdefault('CONVERSATION_LOG_DIR', os.path.join(user_dir, 'conversation_logs'))

        # Bundled resource paths
        os.environ.setdefault('SYSTEM_PROMPT_PATH', get_resource_path('core/system_prompt.txt'))
        os.environ.setdefault('CONFIG_PATH', get_resource_path('config/config.yaml'))

        # External data paths (optional)
        external_dir = get_external_data_dir()
        if external_dir:
            os.environ.setdefault('WIKI_FAISS_PATH', os.path.join(external_dir, 'wiki_faiss'))
            os.environ.setdefault('WIKI_DUMP_PATH', os.path.join(external_dir, 'wiki_dump'))
            os.environ.setdefault('SEM_INDEX_PATH', os.path.join(external_dir, 'semantic_index'))
        else:
            # Disable Wikipedia features if external data not present
            os.environ.setdefault('WIKI_ENABLED', 'false')
            os.environ.setdefault('PROMPT_MAX_WIKI', '0')

        # Huggingface cache - use bundled models
        hf_cache = os.path.join(get_app_dir(), '_internal', 'models')
        if os.path.exists(hf_cache):
            os.environ.setdefault('HF_HOME', hf_cache)
            os.environ.setdefault('TRANSFORMERS_CACHE', hf_cache)
            os.environ.setdefault('HF_HUB_CACHE', hf_cache)
            os.environ.setdefault('HF_HUB_OFFLINE', '1')  # Use bundled models only

    return user_dir


def _load_env_manual(env_path: str) -> None:
    """Manual .env parsing fallback."""
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ.setdefault(key.strip(), value)
    except Exception as e:
        print(f"[Bootstrap] Warning: Could not load .env: {e}")


# =============================================================================
# SPLASH SCREEN (PyInstaller native)
# =============================================================================

def close_splash() -> None:
    """Close PyInstaller splash screen if present."""
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass  # No splash screen in development mode


def update_splash(text: str) -> None:
    """Update splash screen text if present."""
    try:
        import pyi_splash
        pyi_splash.update_text(text)
    except ImportError:
        # In development, print to console
        if not IS_FROZEN:
            print(f"[Startup] {text}")


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize() -> Dict[str, Any]:
    """
    Complete bootstrap initialization.

    Call this at the very start of main.py, after freeze_support().

    Returns:
        dict: Paths and configuration for the application
    """
    # Setup environment (loads .env, sets paths)
    user_dir = setup_environment()

    return {
        'user_data_dir': user_dir,
        'app_dir': get_app_dir(),
        'is_frozen': IS_FROZEN,
        'platform': 'windows' if IS_WINDOWS else 'macos' if IS_MACOS else 'linux',
    }


# =============================================================================
# MODULE-LEVEL EXPORTS
# =============================================================================

# These are available immediately on import
USER_DATA_DIR = get_user_data_dir()
APP_DIR = get_app_dir()


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("BOOTSTRAP MODULE SELF-TEST")
    print("=" * 60)

    print(f"\nIS_FROZEN: {IS_FROZEN}")
    print(f"IS_WINDOWS: {IS_WINDOWS}")
    print(f"IS_MACOS: {IS_MACOS}")
    print(f"IS_LINUX: {IS_LINUX}")

    print(f"\nAPP_DIR: {APP_DIR}")
    print(f"USER_DATA_DIR: {USER_DATA_DIR}")

    print(f"\nget_resource_path('core/system_prompt.txt'):")
    res_path = get_resource_path('core/system_prompt.txt')
    print(f"  Path: {res_path}")
    print(f"  Exists: {os.path.exists(res_path)}")

    print(f"\nget_external_data_dir(): {get_external_data_dir()}")

    print("\nRunning initialize()...")
    config = initialize()
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Self-test complete")
    print("=" * 60)
