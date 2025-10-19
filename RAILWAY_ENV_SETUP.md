
# Railway Environment Variable Setup

## Required: Add ScrapingBee API Key

### Step 1: Go to Railway Dashboard
https://railway.app/dashboard

### Step 2: Select Your Project
Click on "fathom-prospector-api" project

### Step 3: Go to Variables
1. Click on the service (backend)
2. Click on "Variables" tab
3. Click "+ New Variable"

### Step 4: Add ScrapingBee Key
```
Variable Name:  SCRAPINGBEE_API_KEY
Variable Value: 5DYME2AKNFV9MU8CNKBU89SY7S9TZIL7UH73J9Q88KIIEA72EZF9Y6OEQ56A69ZC6A1ET93IQQ1NQWKZ
```

### Step 5: Save and Redeploy
Railway will automatically redeploy after adding the variable.

---

## Verification

After deployment, check the logs for:
```
✅ ScrapingBee enabled for JavaScript rendering
```

If you see:
```
⚠️  ScrapingBee not available - JavaScript rendering disabled
```

Then the API key was not configured correctly.

---

## ScrapingBee Account Info

**Dashboard:** https://app.scrapingbee.com/  
**API Key:** Already provided above  
**Free Tier:** 1,000 requests/month  
**Usage Tracking:** View in ScrapingBee dashboard

---

## Testing

After deployment with API key configured:

1. Run a search in Killeen, TX for dermatologists
2. Check logs for "🐝 Scraping with ScrapingBee"
3. Export CSV and verify First Name, Last Name, Email are populated
4. No "Executable doesn't exist" errors should appear
