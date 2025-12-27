#!/bin/bash
# Launch script for the simplified OCR demo (Module 5 only) - Unix version
#
# This script launches a Streamlit app that demonstrates OCR extraction
# on pre-rectified container ID images (standalone mode).
#
# Usage:
#   bash launch_simple.sh
#   # or
#   ./launch_simple.sh  (after chmod +x)

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="${SCRIPT_DIR}/app.py"

# Verify app exists
if [ ! -f "$APP_PATH" ]; then
    echo "❌ Error: App file not found at $APP_PATH"
    exit 1
fi

# Get port from Python config
PORT=$(python -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}/../..'); from demos.ports_config import get_port; print(get_port('ocr'))")

echo "🚀 Launching OCR Demo (Standalone Mode)..."
echo "📂 App location: $APP_PATH"
echo "🔌 Port: $PORT"
echo "🌐 URL: http://localhost:$PORT"
echo "------------------------------------------------------------"

# Launch Streamlit
python -m streamlit run "$APP_PATH" \
    --server.port="$PORT" \
    --server.headless=true
