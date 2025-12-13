"""PyInstaller hook for spacy.

spaCy needs its language data to be bundled.
The en_core_web_sm model is installed separately.
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# Collect spacy data files
datas = collect_data_files('spacy')

# Try to collect the English model
try:
    datas += collect_data_files('en_core_web_sm')
except Exception:
    pass

# Try to find spacy model from the package
try:
    import spacy
    import en_core_web_sm
    model_path = en_core_web_sm.__path__[0]
    if os.path.isdir(model_path):
        datas.append((model_path, 'en_core_web_sm'))
except Exception:
    pass

# Hidden imports
hiddenimports = collect_submodules('spacy')
hiddenimports += [
    'en_core_web_sm',
    'spacy.lang.en',
]
