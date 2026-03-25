#!/bin/bash
cd "$(dirname "$0")/.."
source venv/bin/activate
export DISPLAY=:99
pkill -f "Xvfb :99" 2>/dev/null
Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
sleep 1
python -m pytest tests/ "$@"
EXIT=$?
pkill -f "Xvfb :99" 2>/dev/null
exit $EXIT
