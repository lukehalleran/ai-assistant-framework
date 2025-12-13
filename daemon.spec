# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Daemon RAG Agent.

Build command: pyinstaller daemon.spec --clean

Output: dist/Daemon/ folder containing:
  - Daemon (or Daemon.exe on Windows)
  - _internal/ (Python runtime and packages)
  - assets/ (icons, splash)

Key decisions:
- ONE-DIR MODE: Prevents 30s startup delay of one-file mode
- NO UPX: Disabled to avoid breaking torch/numpy DLLs
- EXTERNAL DATA NOT BUNDLED: Wiki/FAISS data (100GB+) is optional

Based on audit results (2025-12-12):
- Total startup: ~12-17s (splash screen essential)
- Heaviest imports: sentence_transformers (4.1s), torch (1.8s)
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_all,
)

block_cipher = None

# =============================================================================
# PROJECT ROOT
# =============================================================================

# Get the spec file directory (project root)
spec_dir = os.path.dirname(os.path.abspath(SPEC))

# =============================================================================
# DATA FILES TO BUNDLE
# =============================================================================

datas = [
    # Core application files
    ('core/system_prompt.txt', 'core'),
    ('config/config.yaml', 'config'),

    # Assets
    ('assets/daemon_icon.ico', 'assets'),
    ('assets/daemon_icon.png', 'assets'),
    ('assets/splash.png', 'assets'),

    # Personality configs (if directory exists)
]

# Add personality directory if it exists
personality_dir = os.path.join(spec_dir, 'personality')
if os.path.isdir(personality_dir):
    datas.append(('personality', 'personality'))

# Collect package data files
# spaCy model - this may not work if installed via spacy download
# We'll handle this in hooks
try:
    datas += collect_data_files('en_core_web_sm')
except Exception:
    print("Warning: Could not collect en_core_web_sm data files")

# Sentence transformers
try:
    datas += collect_data_files('sentence_transformers')
except Exception:
    print("Warning: Could not collect sentence_transformers data files")

# Tiktoken encoding files
try:
    datas += collect_data_files('tiktoken')
except Exception:
    print("Warning: Could not collect tiktoken data files")

try:
    datas += collect_data_files('tiktoken_ext', include_py_files=True)
except Exception:
    pass

# Gradio templates and static files (include_py_files for runtime introspection)
try:
    datas += collect_data_files('gradio', include_py_files=True)
except Exception:
    print("Warning: Could not collect gradio data files")

# Gradio client (types.json and other data files)
try:
    datas += collect_data_files('gradio_client')
except Exception:
    print("Warning: Could not collect gradio_client data files")

# safehttpx (version.txt)
try:
    datas += collect_data_files('safehttpx')
except Exception:
    pass

# groovy (version.txt)
try:
    datas += collect_data_files('groovy')
except Exception:
    pass

# Other gradio dependencies that may need data files
for pkg in ['tomlkit', 'ruff', 'httpcore', 'anyio', 'starlette', 'fastapi', 'uvicorn', 'orjson', 'aiofiles']:
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# ChromaDB migrations and configs
try:
    datas += collect_data_files('chromadb')
except Exception:
    print("Warning: Could not collect chromadb data files")

# Transformers
try:
    datas += collect_data_files('transformers')
except Exception:
    pass

# =============================================================================
# HIDDEN IMPORTS
# =============================================================================

hiddenimports = [
    # Multiprocessing (CRITICAL for Windows)
    'multiprocessing',
    'multiprocessing.pool',
    'multiprocessing.process',
    'multiprocessing.spawn',
    'multiprocessing.popen_spawn_win32',
    'multiprocessing.popen_spawn_posix',
    'multiprocessing.reduction',

    # ChromaDB and dependencies
    'chromadb',
    'chromadb.config',
    'chromadb.api',
    'chromadb.api.segment',
    'chromadb.db',
    'chromadb.db.impl',
    'chromadb.segment',
    'chromadb.segment.impl',
    'hnswlib',
    'posthog',

    # Sentence transformers
    'sentence_transformers',
    'sentence_transformers.models',
    'sentence_transformers.util',
    'sentence_transformers.cross_encoder',

    # spaCy
    'spacy',
    'spacy.lang.en',
    'en_core_web_sm',

    # Tiktoken
    'tiktoken',
    'tiktoken_ext',
    'tiktoken_ext.openai_public',

    # Gradio
    'gradio',
    'gradio.themes',
    'gradio.components',
    'gradio.blocks',

    # PyTorch
    'torch',
    'torch.nn',
    'torch.utils',
    'torch._C',

    # Transformers
    'transformers',
    'transformers.models',

    # Encoding
    'encodings.idna',
    'encodings.utf_8',
    'encodings.ascii',
    'encodings.latin_1',

    # Async
    'asyncio',
    'aiohttp',
    'uvicorn',
    'fastapi',

    # Pydantic
    'pydantic',
    'pydantic.fields',
    'pydantic_core',
    'pydantic_core._pydantic_core',

    # HTTP clients
    'httpx',
    'httpcore',
    'anyio',
    'anyio._backends',
    'anyio._backends._asyncio',

    # NumPy internals
    'numpy.core._multiarray_umath',
    'numpy.core._dtype_ctypes',

    # pkg_resources
    'pkg_resources.py2_warn',
    'pkg_resources._vendor',

    # Our modules
    'utils.bootstrap',
    'utils.startup',
    'config.app_config',
    'core.orchestrator',
    'gui.launch',
    'gui.wizard',
    'memory.user_profile',
    'memory.memory_coordinator',
]

# Collect all submodules from complex packages
try:
    hiddenimports += collect_submodules('chromadb')
except Exception:
    pass

try:
    hiddenimports += collect_submodules('gradio')
except Exception:
    pass

try:
    hiddenimports += collect_submodules('sentence_transformers')
except Exception:
    pass

try:
    hiddenimports += collect_submodules('tiktoken')
except Exception:
    pass

try:
    hiddenimports += collect_submodules('transformers')
except Exception:
    pass

# =============================================================================
# EXCLUDES (reduce bundle size, prevent accidental inclusion)
# =============================================================================

excludes = [
    # Test frameworks (keep unittest - scipy/numpy need it at import time)
    'pytest',
    '_pytest',

    # Development tools
    'IPython',
    'jupyter',
    'notebook',

    # Unused GUI frameworks
    'tkinter',
    '_tkinter',
    'PyQt5',
    'PyQt6',
    'PySide2',
    'PySide6',

    # Visualization (not needed for core functionality)
    'matplotlib',
    'plotly',
    'seaborn',

    # Large optional ML frameworks
    'tensorflow',
    'jax',
    'flax',

    # CUDA (we use CPU for portability)
    # Uncomment these if you want CPU-only build
    # 'torch.cuda',
    # 'torch.backends.cudnn',
]

# =============================================================================
# RUNTIME HOOK
# =============================================================================

# Create runtime hook content
runtime_hook_content = '''
"""
Runtime hook - executes before main.py imports.
Sets up environment for frozen executable.
"""
import os
import sys

# Set flag for frozen detection
os.environ['DAEMON_FROZEN'] = '1'

# Ensure multiprocessing uses spawn on all platforms
if sys.platform == 'win32':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
'''

# Write runtime hook
runtime_hook_path = os.path.join(spec_dir, 'hooks', 'runtime_hook.py')
os.makedirs(os.path.dirname(runtime_hook_path), exist_ok=True)
with open(runtime_hook_path, 'w') as f:
    f.write(runtime_hook_content)

# =============================================================================
# ANALYSIS
# =============================================================================

a = Analysis(
    ['main.py'],
    pathex=[spec_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],  # Custom hooks directory
    hooksconfig={},
    runtime_hooks=[runtime_hook_path],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# =============================================================================
# PYZ (Python bytecode archive)
# =============================================================================

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# =============================================================================
# SPLASH SCREEN (shows immediately during Python startup)
# =============================================================================

# Note: Splash requires PyInstaller 5.0+
splash = None
splash_file = os.path.join(spec_dir, 'assets', 'splash.png')
if os.path.exists(splash_file):
    try:
        splash = Splash(
            splash_file,
            binaries=a.binaries,
            datas=a.datas,
            text_pos=(30, 250),  # Position for loading text
            text_size=12,
            text_color='#666666',
            minify_script=True,
            always_on_top=True,
        )
    except Exception as e:
        print(f"Warning: Could not create splash screen: {e}")
        splash = None

# =============================================================================
# EXECUTABLE
# =============================================================================

exe_args = [
    pyz,
    a.scripts,
]

if splash:
    exe_args.append(splash)
    exe_args.append(splash.binaries)

exe_args.extend([
    [],  # Don't include binaries in EXE (one-dir mode)
])

exe_kwargs = {
    'exclude_binaries': True,  # CRITICAL: Enable one-dir mode
    'name': 'Daemon',
    'debug': False,
    'bootloader_ignore_signals': False,
    'strip': False,
    'upx': False,  # CRITICAL: Disabled - breaks torch/numpy DLLs
    'console': False,  # GUI app, no console window
    'disable_windowed_traceback': False,
    'argv_emulation': False,
    'target_arch': None,
    'codesign_identity': None,
    'entitlements_file': None,
}

# Platform-specific settings
icon_file = os.path.join(spec_dir, 'assets', 'daemon_icon.ico')
if os.path.exists(icon_file):
    exe_kwargs['icon'] = icon_file

# Windows version info (if file exists)
version_file = os.path.join(spec_dir, 'file_version_info.txt')
if sys.platform == 'win32' and os.path.exists(version_file):
    exe_kwargs['version'] = version_file

# macOS bundle identifier
if sys.platform == 'darwin':
    exe_kwargs['bundle_identifier'] = 'com.daemon.ragagent'

exe = EXE(
    *exe_args,
    **exe_kwargs,
)

# =============================================================================
# COLLECT (creates the one-dir folder structure)
# =============================================================================

coll_args = [
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
]

# Note: Splash data is automatically included via the EXE, no need to add separately

coll = COLLECT(
    *coll_args,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Daemon',
)

# =============================================================================
# macOS APP BUNDLE (optional)
# =============================================================================

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Daemon.app',
        icon='assets/daemon_icon.icns' if os.path.exists('assets/daemon_icon.icns') else None,
        bundle_identifier='com.daemon.ragagent',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.13',
        },
    )
