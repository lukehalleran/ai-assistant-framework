
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
