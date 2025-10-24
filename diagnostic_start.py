#!/usr/bin/env python3
"""
Diagnostic startup script that logs everything before starting the server.
This will help identify why Deploy Logs are blank.
"""

import sys
import os

print("=" * 60)
print("🔍 DIAGNOSTIC STARTUP SCRIPT")
print("=" * 60)

# Step 1: Environment Check
print("\n1️⃣ Checking environment...")
print(f"   Python version: {sys.version}")
print(f"   Working directory: {os.getcwd()}")
print(f"   PORT variable: {os.environ.get('PORT', 'NOT SET')}")
print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")

# Step 2: Check if files exist
print("\n2️⃣ Checking required files...")
required_files = ['api_server.py', 'prospect.py', 'requirements.txt']
for file in required_files:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")

# Step 3: Test imports one by one
print("\n3️⃣ Testing imports...")

try:
    print("   Importing sys, os... ", end="")
    import sys, os
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("   Importing asyncio... ", end="")
    import asyncio
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("   Importing aiohttp... ", end="")
    import aiohttp
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("   Importing fastapi... ", end="")
    from fastapi import FastAPI
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("   Importing uvicorn... ", end="")
    import uvicorn
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

try:
    print("   Importing api_server module... ", end="")
    import api_server
    print("✅")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\n   Full traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Start the server
print("\n4️⃣ All checks passed! Starting server...")
print("=" * 60)

port = int(os.environ.get('PORT', 8000))
print(f"🚀 Starting uvicorn on port {port}...")

uvicorn.run(
    "api_server:app",
    host="0.0.0.0",
    port=port,
    timeout_keep_alive=300
)
