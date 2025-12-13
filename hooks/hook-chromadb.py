"""PyInstaller hook for chromadb.

ChromaDB has many dynamically imported modules and data files.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files (migrations, etc.)
datas = collect_data_files('chromadb')

# Collect all submodules (many are dynamically imported)
hiddenimports = collect_submodules('chromadb')

# ChromaDB also needs hnswlib
hiddenimports += ['hnswlib']

# PostHog telemetry (optional but chromadb imports it)
hiddenimports += ['posthog']
