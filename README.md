
# Fathom Prospector - Python API Server

This is the standalone Python API server that handles prospecting searches for the Fathom web app.

## Quick Start

### 1. Setup Environment

```bash
cd python-api

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

Required API keys:
- `FATHOM_API_KEY` - Secure key for authenticating Next.js app (generate a random string)
- `GOOGLE_PLACES_API_KEY` - Your Google Places API key  
- `GEMINI_API_KEY` - Your Google Gemini AI API key

### 2. Start Server

```bash
# Make start script executable
chmod +x start.sh

# Start the server
./start.sh
```

The server will start on http://localhost:8000

### 3. Test Server

Open http://localhost:8000/health in your browser. You should see:
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

## API Endpoints

### Health Check
```
GET /health
```

### Start Search
```
POST /api/prospect/search
Headers:
  X-API-Key: your-fathom-api-key
  Content-Type: application/json
Body:
{
  "keywords": ["cosmetic surgery", "aesthetic surgery"],
  "location": "Beverly Hills, CA",
  "radius": 15,
  "maxResults": 150
}
```

### Check Search Status
```
GET /api/prospect/status/{job_id}
Headers:
  X-API-Key: your-fathom-api-key
```

### Get Search Results
```
GET /api/prospect/results/{job_id}
Headers:
  X-API-Key: your-fathom-api-key
```

## Deployment Options

### Option A: Local Machine (Testing)

1. Start the API server on your machine: `./start.sh`
2. Use ngrok to expose it publicly:
   ```bash
   # Install ngrok
   brew install ngrok  # Mac
   # or download from ngrok.com

   # Expose your local server
   ngrok http 8000
   ```
3. Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)
4. Update your Next.js app environment variable:
   ```
   PYTHON_API_URL=https://abc123.ngrok.io
   ```

**Pros**: Quick setup, free  
**Cons**: Requires your machine to stay on, URL changes each restart

### Option B: Cloud VPS (Production)

Deploy to a cloud server for 24/7 uptime:

**DigitalOcean Example** ($5/month):
```bash
# On your VPS
git clone your-repo
cd python-api
./start.sh

# Set up systemd service for auto-restart
sudo nano /etc/systemd/system/fathom-api.service
```

**Systemd service file**:
```ini
[Unit]
Description=Fathom Prospector API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/python-api
ExecStart=/home/ubuntu/python-api/venv/bin/python api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable fathom-api
sudo systemctl start fathom-api

# Check status
sudo systemctl status fathom-api
```

### Option C: AWS Lambda (Serverless)

For advanced users - deploy as serverless function with API Gateway.

## Security

- The `FATHOM_API_KEY` prevents unauthorized access
- Generate a strong random key: `openssl rand -hex 32`
- Only your Next.js app should know this key
- Add your production domain to CORS origins in `api_server.py`

## Monitoring

Check server logs:
```bash
# If running in terminal
# Logs appear in real-time

# If running as systemd service
sudo journalctl -u fathom-api -f
```

## Troubleshooting

**"Python not found"**
- Install Python 3: `sudo apt install python3 python3-pip python3-venv`

**"Module not found"**
- Install dependencies: `pip install -r requirements.txt`

**"Health check fails"**
- Check `.env` file has correct API keys
- Verify prospect.py exists in `../app/scripts/`

**"Connection refused"**
- Check server is running: `ps aux | grep api_server`
- Check firewall allows port 8000: `sudo ufw allow 8000`

## Support

For issues, check:
1. Server logs
2. Health check endpoint
3. Environment variables
4. Python version (3.8+ required)
