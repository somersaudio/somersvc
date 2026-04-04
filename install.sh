#!/bin/bash
# SomerSVC One-Line Installer
# Usage: curl -sL https://raw.githubusercontent.com/somersaudio/somersvc/main/install.sh | bash

set -e

echo "================================"
echo "  SomerSVC - Installing..."
echo "================================"
echo ""

# Install Homebrew if needed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python 3.10 for RVC support
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10 (for RVC support)..."
    brew install python@3.10
fi

# Install Python 3 if needed
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    brew install python3
fi

# Clone the repo
INSTALL_DIR="$HOME/Desktop/SomerSVC"
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull --ff-only origin main
else
    echo "Downloading SomerSVC..."
    git clone https://github.com/somersaudio/somersvc.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Run setup
bash setup.sh

echo ""
echo "================================"
echo "  SomerSVC installed!"
echo "  Double-click SomerSVC on your Desktop to launch."
echo "================================"
