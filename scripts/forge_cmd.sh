#!/bin/bash
# Usage: forge_cmd.sh "command" [screenshot_name]
# Sends command to Forge GUI and optionally takes screenshot
export DISPLAY=:99
CMD="$1"
SHOT="${2:-}"

WID=$(xdotool search --name "Forge" 2>/dev/null | head -1)
if [ -z "$WID" ]; then
    echo "ERROR: Forge not running"
    exit 1
fi

# Focus command widget
xdotool windowactivate --sync $WID 2>/dev/null
sleep 0.2
xdotool key ctrl+0
sleep 0.2

# Type command fast (10ms delay instead of 30ms)
xdotool type --delay 10 "$CMD"
xdotool key Return

if [ -n "$SHOT" ]; then
    # Wait for command to execute then screenshot
    WAIT="${3:-5}"
    sleep $WAIT
    scrot /tmp/forge_${SHOT}.png
    echo "SCREENSHOT:/tmp/forge_${SHOT}.png"
else
    echo "CMD_SENT"
fi
