#!/bin/bash
# DVC Setup Script for Container ID Research Project
# This script initializes DVC and configures the Google Drive remote

set -e  # Exit on error

echo "=================================="
echo "DVC Setup Script"
echo "=================================="

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "❌ Error: DVC is not installed"
    echo "Please install DVC first: pip install dvc[gdrive]"
    exit 1
fi

echo "✓ DVC is installed"

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    echo "✓ DVC initialized"
else
    echo "✓ DVC already initialized"
fi

# Configure Google Drive remote
echo ""
echo "Configuring Google Drive remote..."
dvc remote modify storage gdrive_use_service_account false
echo "✓ Remote configured"

# Check remote status
echo ""
echo "Current DVC configuration:"
dvc config --list

echo ""
echo "=================================="
echo "✓ DVC Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Run 'dvc pull' to download data from Google Drive"
echo "2. You may need to authenticate with Google Drive on first pull"
echo ""

