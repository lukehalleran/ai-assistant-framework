"""PyInstaller hook for gradio.

Gradio has many static files and submodules.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Gradio has many static files
datas = collect_data_files('gradio')

# Include all submodules
hiddenimports = collect_submodules('gradio')

# Also need fastapi/starlette
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('starlette')

# Uvicorn for serving
hiddenimports += collect_submodules('uvicorn')
