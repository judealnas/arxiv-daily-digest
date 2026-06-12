#!/usr/bin/env python3
"""Wrapper: runs arxiv-catchup with quant-ph --yesterday."""
import subprocess, sys
script = "/home/jalnas/.hermes/skills/research/arxiv-catchup/scripts/arxiv-catchup.py"
# Hardcode args because the cron system passes script+args as a single path string.
sys.exit(subprocess.call([sys.executable, script, "quant-ph", "--yesterday"]))
