#!/bin/bash
# MetaPipe Results Viewer - Quick Access Script

echo "========================================================================"
echo "MetaPipe Results Viewer"
echo "========================================================================"
echo ""
echo "Opening results dashboard in your browser..."
echo ""

# Open HTML dashboard
open RESULTS/index.html

echo "✅ Dashboard opened!"
echo ""
echo "You can also view results directly:"
echo "  📊 Consolidated Report: cat RESULTS/CONSOLIDATED_REPORT.md"
echo "  📈 Figures: open RESULTS/Figure_*.png"
echo "  📋 Tables: cat RESULTS/Table_*.csv"
echo ""
echo "========================================================================"
