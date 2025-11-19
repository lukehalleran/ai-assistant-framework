#!/usr/bin/env bash
ls -la gui; echo '---'; nl -ba gui/launch.py | sed -n '1,260p'
