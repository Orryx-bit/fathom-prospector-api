
# 🚀 DEPLOY RIGHT NOW - Step by Step

## Secrets Checklist (Provide Your Own Values)
✅ Google Maps API: `<store-in-your-secrets-manager>`
✅ Google Places API: `<store-in-your-secrets-manager>`
✅ Gemini API: `<store-in-your-secrets-manager>`

---

## Step 1: Download This Entire Folder
Download `/home/ubuntu/fathom-railway-fresh-deploy` to your computer

All files include placeholder secrets—replace them with your own keys before committing or deploying.

---

## Step 2: Create New GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `fathom-v1`
3. Make it **Public** or **Private** (your choice)
4. **DO NOT** check any boxes (no README, no .gitignore, no license)
5. Click **"Create repository"**

---

## Step 3: Upload Files to GitHub

### Option A: Drag and Drop (Easiest)
1. On your new GitHub repository page
2. Click **"uploading an existing file"**
3. Drag and drop ALL files from the downloaded folder (replace any placeholder `.env` secrets with your own values before committing)
4. **MAKE SURE** the `scoring_system` folder with all 7 files inside is included
5. Scroll down and click **"Commit changes"**

### Option B: Command Line
```bash
cd /path/to/downloaded/fathom-railway-fresh-deploy
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fathom-v1.git
git push -u origin main
```

---

## Step 4: Verify Files on GitHub

Go to your repository on GitHub and verify you see:
- ✅ `.gitignore`
- ✅ `.env.example`
- ✅ `README.md`
- ✅ `api_server.py`
- ✅ `prospect.py`
- ✅ `requirements.txt`
- ✅ `railway.json`
- ✅ **`scoring_system/` folder** (click it to see all 7 files inside)

---

## Step 5: Deploy to Railway

1. Go to: https://railway.app
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository: `fathom-v1`
5. Railway will detect the configuration automatically

---

## Step 6: Add Environment Variables on Railway

In the Railway dashboard:
1. Click on your service
2. Go to **"Variables"** tab
3. Click **"Add Variable"** and populate the following keys with freshly generated or rotated values (never commit real keys to git):

```
GOOGLE_MAPS_API_KEY=<google-maps-api-key>
```
```
GOOGLE_PLACES_API_KEY=<google-places-api-key>
```
```
GEMINI_API_KEY=<gemini-api-key>
```
```
FATHOM_API_KEY=<generate-unique-shared-secret>
```
```
PORT=8000
```
```
HOST=0.0.0.0
```

Keep these secrets in your cloud provider's secret manager or Railway environment variables, and rotate them if you suspect exposure.

---

## Step 7: Wait for Deployment

Railway will:
1. Install all dependencies
2. Start the server
3. Provide a public URL (something like `https://your-app.up.railway.app`)

This takes 3-5 minutes

---

## Step 8: Test Your API

Once deployed, open:
```
https://your-railway-url.up.railway.app/health
```

You should see:
```json
{
  "status": "healthy",
  "checks": {
    "python": true,
    "prospect_script": true,
    "google_api": true,
    "gemini_api": true
  },
  "metrics": {...}
}
```

---

## Step 9: Update Your Web App

Update your web app's API URL to the new Railway URL.

---

## ✅ You're Done!

Your API is now running 24/7 on Railway with all the scoring_system files included!

---

## Troubleshooting

If you see errors in Railway logs:

1. **Check Environment Variables**: Make sure all 6 variables are set
2. **Check Files**: Verify scoring_system folder uploaded correctly
3. **View Logs**: Click "View Logs" in Railway to see what's failing

---

## Need Help?

If deployment fails, send me:
1. Screenshot of your GitHub repository files
2. Screenshot of Railway logs
3. The error message

I'll help you fix it immediately!
