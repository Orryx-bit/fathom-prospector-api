
# Fathom Prospector API - Production Deployment

Backend API for Fathom Prospector medical device prospecting system.

## ✅ Deployment Checklist

### 1. Create New GitHub Repository
```bash
1. Go to https://github.com/new
2. Name: fathom-v1 (or any name you prefer)
3. Privacy: Public or Private (your choice)
4. DO NOT initialize with README, .gitignore, or license
5. Click "Create repository"
```

### 2. Upload All Files to GitHub
```bash
# Upload these files to your new GitHub repository:
├── .gitignore
├── .env.example
├── README.md
├── api_server.py
├── prospect.py
├── requirements.txt
├── railway.json
└── scoring_system/
    ├── __init__.py
    ├── data_generator.py
    ├── evaluation.py
    ├── feature_engineering.py
    ├── pipeline.py
    ├── scoring_engine.py
    └── venus_adapter.py
```

**IMPORTANT**: Upload the ENTIRE scoring_system folder with all 7 files inside

### 3. Deploy to Railway

#### Step 1: Create New Project
1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your new repository (fathom-v1)

#### Step 2: Configure Environment Variables
In Railway dashboard, add these variables using values issued from your secrets manager (do not reuse production keys across environments):

```
GOOGLE_MAPS_API_KEY=<google-maps-api-key>
GOOGLE_PLACES_API_KEY=<google-places-api-key>
GEMINI_API_KEY=<gemini-api-key>
FATHOM_API_KEY=<generate-unique-shared-secret>
PORT=8000
HOST=0.0.0.0
```

Create a local `.env` file by copying `.env.example` and substituting the placeholders above with freshly generated credentials. Rotate secrets regularly and revoke any keys that may have been exposed.

#### Step 3: Deploy
1. Railway will automatically detect the configuration
2. Click "Deploy"
3. Wait for build to complete (3-5 minutes)
4. Railway will provide a public URL

### 4. Test the Deployment

Once deployed, test the health endpoint:
```
https://your-railway-url.up.railway.app/health
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

### 5. Update Web App

Update your Next.js web app's API URL to point to the new Railway URL.

## API Endpoints

- `GET /health` - Health check
- `POST /api/prospect/search` - Start new prospect search
- `GET /api/prospect/status/{job_id}` - Check search status

## Support

If deployment fails, check Railway logs for errors.
