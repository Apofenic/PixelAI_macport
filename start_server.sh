#!/usr/bin/env bash

# Local AI Generator - macOS/Linux Setup Script
# Mirrors Start Server.bat behavior for Unix-like systems

set -o pipefail
set -u

die() { echo "✗ $*" >&2; exit 1; }

echo
echo "========================================================"
echo "   Local AI Generator for Aseprite - Auto Setup (Unix)"
echo "========================================================"
echo

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || die "Failed to cd into script directory"

echo "Current directory: $SCRIPT_DIR"

echo "[1/4] Checking required files..."
if [[ ! -f "startup_script.py" ]]; then
  die "startup_script.py not found in current directory. Please ensure all files are in the same folder"
fi

if [[ ! -f "sd_server.py" ]]; then
  if [[ -f "paste.txt" ]]; then
    echo "Found paste.txt - copying to sd_server.py..."
    cp -f "paste.txt" "sd_server.py" || die "Failed to copy paste.txt to sd_server.py"
    echo "✓ Created sd_server.py from paste.txt"
  else
    die "Neither sd_server.py nor paste.txt found"
  fi
else
  echo "✓ sd_server.py found"
fi

echo "[2/4] Checking Python installation..."
PY_CMD=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    PY_CMD="$c"
    break
  fi
done

if [[ -z "$PY_CMD" ]]; then
  cat >&2 <<'MSG'
✗ Python not found.

PLEASE INSTALL PYTHON MANUALLY:
1. Go to https://python.org
2. Download Python 3.8 or newer
3. On macOS, prefer the official installer or use Homebrew: brew install python
4. Re-open your terminal after installation
5. Run this script again
MSG
  exit 1
fi

"$PY_CMD" --version || die "Unable to execute $PY_CMD"

# Optional: enforce Python >= 3.8
if ! "$PY_CMD" -c 'import sys; exit(0 if sys.version_info >= (3,8) else 1)'; then
  die "Python 3.8+ is required"
fi
echo "✓ Python is available"

echo "[3/4] Setting up virtual environment..."
if [[ -f "venv/bin/activate" ]]; then
  echo "✓ Virtual environment already exists"
else
  echo "Creating virtual environment..."
  "$PY_CMD" -m venv venv || die "Failed to create virtual environment"
  echo "✓ Virtual environment created"
fi

echo "[4/4] Setting up directories..."
mkdir -p loras models || die "Failed to create required directories"
echo "✓ Directories created"

echo
echo "========================================================"
echo "             Basic Setup Complete!"
echo "========================================================"
echo
echo "Now starting Python setup and server..."
echo "Python will handle dependency installation and server startup."
echo

# Activate virtual environment and start Python setup
# shellcheck disable=SC1091
source "venv/bin/activate" || die "Failed to activate virtual environment"
echo "✓ Virtual environment activated"
echo "Starting Python setup script..."
echo

"$PY_CMD" startup_script.py
STATUS=$?

echo
echo "========================================================"
echo "                  SETUP/SERVER ENDED"
echo "========================================================"
echo

exit $STATUS
