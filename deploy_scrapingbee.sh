#!/bin/bash

echo "🐝 ScrapingBee Deployment Script"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "prospect.py" ]; then
    echo "❌ Error: Not in fathom-api-github directory"
    exit 1
fi

# Check if ScrapingBee integration file exists
if [ ! -f "scrapingbee_integration.py" ]; then
    echo "❌ Error: scrapingbee_integration.py not found"
    exit 1
fi

echo "📋 Files to be deployed:"
echo "   ✅ prospect.py (ScrapingBee integration)"
echo "   ✅ scrapingbee_integration.py (new)"
echo "   ✅ requirements.txt (scrapingbee added)"
echo "   ✅ start.sh (Playwright removed)"
echo "   ✅ SCRAPINGBEE_UPGRADE.md (documentation)"
echo ""

echo "🔍 Syntax check..."
python3 -m py_compile prospect.py || { echo "❌ prospect.py has syntax errors"; exit 1; }
python3 -m py_compile scrapingbee_integration.py || { echo "❌ scrapingbee_integration.py has syntax errors"; exit 1; }
echo "✅ All files pass syntax check"
echo ""

echo "📦 Git status:"
git status --short
echo ""

echo "🚀 Ready to deploy!"
echo ""
echo "Next steps:"
echo "1. Add SCRAPINGBEE_API_KEY to Railway environment variables"
echo "2. Run: git add ."
echo "3. Run: git commit -m 'Replace Playwright with ScrapingBee'"
echo "4. Run: git push origin main"
echo ""
echo "Railway will auto-deploy in ~5 minutes"
