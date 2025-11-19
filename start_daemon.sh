#!/usr/bin/env bash

# Unset any existing semantic search environment variables
unset SEM_INDEX_PATH
unset SEM_META_PATH

# Start the daemon
python main.py
