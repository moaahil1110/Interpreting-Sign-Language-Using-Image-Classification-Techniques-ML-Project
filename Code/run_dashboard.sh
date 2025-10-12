#!/bin/bash
# Quick start script for ASL Dashboard
# Usage: ./run_dashboard.sh

echo "🤟 Starting ASL Alphabet Recognition Dashboard..."
echo ""

# Check if we're in the Code directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the Code directory."
    exit 1
fi

# Check if requirements are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing required packages..."
    pip install -r requirements_dashboard.txt
fi

# Check if model exists
if [ ! -f "../best_asl_model_rtx4060_optimized.pth" ]; then
    echo "⚠️  Warning: Model file not found at ../best_asl_model_rtx4060_optimized.pth"
    echo "Please ensure your model file is in the correct location."
    echo ""
fi

# Launch dashboard
echo "🚀 Launching dashboard..."
echo "📱 Dashboard will open at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run app.py
