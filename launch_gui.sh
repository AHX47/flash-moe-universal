#!/usr/bin/env bash
# Quick GUI launcher
cd "$(dirname "$0")"
python3 gui/app.py "$@" || python gui/app.py "$@"
