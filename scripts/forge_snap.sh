#!/bin/bash
# Quick screenshot - no fuss
export DISPLAY=:99
NAME="${1:-snap}"
scrot /tmp/forge_${NAME}.png
echo "/tmp/forge_${NAME}.png"
