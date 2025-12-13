"""PyInstaller hook for tiktoken.

Tiktoken needs its encoding files to be bundled.
"""
from PyInstaller.utils.hooks import collect_data_files

# Tiktoken needs its encoding files
datas = collect_data_files('tiktoken')

try:
    datas += collect_data_files('tiktoken_ext')
except Exception:
    pass

# Hidden imports for registry
hiddenimports = [
    'tiktoken_ext',
    'tiktoken_ext.openai_public',
]
