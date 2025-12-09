#!/bin/bash
# Run DVC Data Processing Pipeline
# Executes all data preparation stages

set -e  # Exit on error

echo "=================================="
echo "DVC Pipeline Execution"
echo "=================================="

# Check if DVC is initialized
if [ ! -d ".dvc" ]; then
    echo "❌ Error: DVC not initialized"
    echo "Please run scripts/setup_dvc.sh first"
    exit 1
fi

# Check if data exists
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "❌ Error: Raw data not found"
    echo "Please run scripts/download_data.sh first"
    exit 1
fi

# Check for params.yaml
if [ ! -f "params.yaml" ]; then
    echo "❌ Error: params.yaml not found"
    exit 1
fi

echo ""
echo "Starting DVC pipeline..."
echo ""

# Run DVC reproduce to execute all stages
dvc repro

echo ""
echo "=================================="
echo "✓ Pipeline Execution Complete!"
echo "=================================="
echo ""
echo "Generated datasets:"
echo "  - Detection: data/processed/detection/"
echo "  - Localization: data/processed/localization/"
echo ""
echo "Next steps:"
echo "1. Review processed datasets"
echo "2. Start training: python src/detection/train.py --config <config>"
echo ""

