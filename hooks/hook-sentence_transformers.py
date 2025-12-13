"""PyInstaller hook for sentence-transformers.

Sentence transformers depends on transformers and torch.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect data files
datas = collect_data_files('sentence_transformers')

# Collect submodules
hiddenimports = collect_submodules('sentence_transformers')

# Also need transformers
hiddenimports += collect_submodules('transformers')

# Cross-encoder support
hiddenimports += [
    'sentence_transformers.cross_encoder',
    'sentence_transformers.cross_encoder.CrossEncoder',
]
