
# Fathom API - Railway Deployment Guide

## Quick Deploy to Railway

### Prerequisites
- Railway account (https://railway.app)
- This GitHub repository

### Step 1: Connect GitHub to Railway

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose this repository
5. Railway will automatically detect it's a Python project

### Step 2: Configure Environment Variables

In Railway dashboard, add these variables:

```
FATHOM_API_KEY=132237302e415a4246f650e1a2fbefa6b3f780dcac54d0546a59858ad4b2ba01
GOOGLE_PLACES_API_KEY=AIzaSyB3ebRmx1PrJzwvd9f618GEM2ZIewJ0tAM
GEMINI_API_KEY=AIzaSyARwUznisFAVug0XCbd7AFXNamIyzcT5b8
```

### Step 3: Deploy!

Railway will automatically:
- Install dependencies from `requirements.txt`
- Start the API server with `python api_server.py`
- Provide an HTTPS URL

### Health Check

After deployment, test the API:
```
https://your-app.up.railway.app/health
```

Should return:
```json
{
  "status": "healthy",
  "checks": {
    "python": true,
    "prospect_script": true,
    "google_api": true,
    "gemini_api": true
  }
}
```

## Files Included

- `api_server.py` - Main FastAPI server
- `prospect.py` - Prospect search logic
- `scoring_system/` - AI scoring engine
- `config/` - Configuration files
- `requirements.txt` - Python dependencies

## Support

If you encounter issues, check Railway logs for error messages.
