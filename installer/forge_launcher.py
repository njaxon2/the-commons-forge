#!/usr/bin/env python3
"""Forge launcher for PyInstaller bundle."""
import sys
import os

if getattr(sys, 'frozen', False):
    os.environ['FORGE_FROZEN'] = '1'

from forge.__main__ import main
main()
