#!/bin/bash
# Data Download Script
# Pulls data from DVC remote (Google Drive)

set -e  # Exit on error

echo "=================================="
echo "Data Download Script"
echo "=================================="

# Check if DVC is initialized
if [ ! -d ".dvc" ]; then
    echo "❌ Error: DVC not initialized"
    echo "Please run scripts/setup_dvc.sh first"
    exit 1
fi

# Check DVC remote configuration
echo "Checking DVC remote configuration..."
dvc remote list

# Pull data from remote
echo ""
echo "Pulling data from Google Drive..."
echo "Note: You may need to authenticate with Google Drive"
echo ""

dvc pull

echo ""
echo "=================================="
echo "✓ Data Download Complete!"
echo "=================================="
echo ""
echo "Data location:"
echo "  - Raw images: data/raw/"
echo "  - Annotations: data/annotations/"
echo ""
echo "Next steps:"
echo "1. Run 'dvc repro' to process data"
echo "2. Start training models"
echo ""

