#!/bin/bash
# MetaPipe Test Runner Script

echo "======================================================================"
echo "MetaPipe Comprehensive Test Suite"
echo "======================================================================"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest not found. Installing..."
    pip install pytest pytest-cov
fi

echo ""
echo "[1/3] Running Unit Tests..."
echo "----------------------------------------------------------------------"
pytest tests/ -v --tb=short --disable-warnings

echo ""
echo "[2/3] Running Tests with Coverage..."
echo "----------------------------------------------------------------------"
pytest tests/ --cov=metapipe --cov-report=term-missing --cov-report=html --disable-warnings

echo ""
echo "[3/3] Generating Test Report..."
echo "----------------------------------------------------------------------"
pytest tests/ --html=test_report.html --self-contained-html --disable-warnings

echo ""
echo "======================================================================"
echo "✅ Test Suite Complete!"
echo "======================================================================"
echo "Coverage report: htmlcov/index.html"
echo "Test report: test_report.html"
echo "======================================================================"
