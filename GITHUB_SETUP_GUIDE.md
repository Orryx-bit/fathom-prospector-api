
# 🚀 GitHub Setup Guide for Fathom API

Follow these steps to get your API on GitHub and connected to Railway.

---

## Step 1: Create a GitHub Account (if you don't have one)

1. Go to https://github.com
2. Click "Sign up"
3. Follow the prompts to create your account

---

## Step 2: Create a New Repository

1. Go to https://github.com/new
2. Fill in the details:
   - **Repository name**: `fathom-api` (or any name you like)
   - **Description**: "Fathom Medical Device Prospector API"
   - **Visibility**: Choose **Private** (recommended) or Public
   - **DON'T** check "Initialize with README" (we already have files)
3. Click **"Create repository"**

---

## Step 3: Upload Your Files to GitHub

GitHub will show you a page with instructions. We'll use the "upload files" method (easiest):

### Option A: Upload via Web Interface (Easiest!)

1. On your new repository page, click **"uploading an existing file"** link
2. Drag and drop ALL these files from your computer:
   - `api_server.py`
   - `prospect.py`
   - `requirements.txt`
   - `README.md`
   - `DEPLOYMENT.md`
   - `.gitignore`
   - `.env.example`
   - `railway.json`
   - **ENTIRE `scoring_system` folder** (drag the whole folder)
   - **ENTIRE `config` folder**
3. Add a commit message: "Initial commit - Fathom API"
4. Click **"Commit changes"**

### Option B: Upload via Git Command Line (if you prefer)

```bash
cd /home/ubuntu/fathom-api-github
git init
git add .
git commit -m "Initial commit - Fathom API"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fathom-api.git
git push -u origin main
```

---

## Step 4: Connect GitHub to Railway

1. Go to https://railway.app
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Click **"Configure GitHub App"** (first time only)
5. Authorize Railway to access your GitHub
6. Select your **`fathom-api`** repository
7. Railway will start deploying automatically!

---

## Step 5: Add Environment Variables in Railway

1. Click on your project in Railway
2. Click on the service (it might say "fathom-api" or similar)
3. Go to **"Variables"** tab
4. Add these three variables:

**Variable 1:**
- Name: `FATHOM_API_KEY`
- Value: `132237302e415a4246f650e1a2fbefa6b3f780dcac54d0546a59858ad4b2ba01`

**Variable 2:**
- Name: `GOOGLE_PLACES_API_KEY`
- Value: `AIzaSyB3ebRmx1PrJzwvd9f618GEM2ZIewJ0tAM`

**Variable 3:**
- Name: `GEMINI_API_KEY`
- Value: `AIzaSyARwUznisFAVug0XCbd7AFXNamIyzcT5b8`

5. Railway will automatically redeploy with the new variables

---

## Step 6: Get Your New API URL

1. In Railway, go to **"Settings"** tab
2. Look for **"Domains"** section
3. You'll see a URL like: `https://fathom-api-production.up.railway.app`
4. **Copy this URL** - you'll need it to update your website!

---

## Step 7: Test Your API

Open this URL in your browser (replace with YOUR Railway URL):
```
https://fathom-api-production.up.railway.app/health
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
  }
}
```

---

## Step 8: Update Your Website

Once your API is working, I'll update your website to use the new GitHub-deployed Railway URL!

---

## 🎉 Benefits of This Setup

- ✅ All files properly uploaded (including `scoring_system`)
- ✅ Automatic deployments when you push to GitHub
- ✅ Version control for your code
- ✅ Professional deployment setup
- ✅ Easy rollbacks if needed

---

## Need Help?

If you get stuck on any step, just send me a screenshot and I'll Computer Usede you through it! 🚀

---

## Files to Download

All your API files are ready in: `/home/ubuntu/fathom-api-github/`

You can download this entire folder and upload to GitHub!
