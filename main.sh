#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$PROJECT_DIR"
$PYTHON_BIN src/runner.py
