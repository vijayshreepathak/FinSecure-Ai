# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy the FinSecure AI dashboard to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. A Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
3. Your project pushed to GitHub (already done ✅)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Your repository is already set up with:
- ✅ `dashboard/app_advanced.py` - Main dashboard file
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Properly configured

### 2. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select your repository: `vijayshreepathak/FinSecure-Ai`
   - Choose branch: `main`
   - Set main file path: `dashboard/app_advanced.py`
   - Click "Deploy!"

### 3. Configure Secrets

After deployment, you need to configure the API URL:

1. **Go to App Settings**
   - Click the three dots (⋮) next to your app
   - Select "Settings"

2. **Add Secrets**
   - Click on "Secrets" tab
   - Add the following:

```toml
API_URL = "https://your-api-url.com"
```

**Important Notes:**
- If your FastAPI backend is also deployed, use that URL
- If running locally, you can use a tunneling service like:
  - **ngrok**: `ngrok http 8000` (provides a public URL)
  - **localtunnel**: `lt --port 8000`
  - **Cloudflare Tunnel**: For production use

### 4. Deploy FastAPI Backend (Optional but Recommended)

For a complete deployment, you should also deploy the FastAPI backend:

#### Option A: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Create new project
3. Connect GitHub repository
4. Deploy from `app/` directory
5. Set environment variables if needed

#### Option B: Deploy to Render
1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### Option C: Deploy to Heroku
1. Create `Procfile`:
   ```
   web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
2. Deploy using Heroku CLI or GitHub integration

#### Option D: Use ngrok for Local Testing
```bash
# Install ngrok
# Run your FastAPI locally
uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal
ngrok http 8000

# Use the ngrok URL in Streamlit secrets
```

### 5. Update Streamlit Secrets

Once your API is deployed, update the Streamlit secrets:

```toml
API_URL = "https://your-deployed-api.railway.app"
# or
API_URL = "https://your-deployed-api.onrender.com"
# or
API_URL = "https://your-ngrok-url.ngrok.io"
```

### 6. Verify Deployment

1. Your Streamlit app will be available at:
   `https://your-app-name.streamlit.app`

2. Test the connection:
   - Open the dashboard
   - Check if it can connect to your API
   - Try making a prediction

## Troubleshooting

### Issue: "Cannot connect to API"
**Solution:**
- Verify API_URL in Streamlit secrets is correct
- Check if your API is publicly accessible
- Ensure CORS is enabled on your FastAPI backend

### Issue: "Module not found"
**Solution:**
- Check `requirements.txt` includes all dependencies
- Verify all packages are listed

### Issue: "API timeout"
**Solution:**
- Your API might be sleeping (free tier limitations)
- Consider using a paid hosting service
- Or use ngrok for development

### Enable CORS on FastAPI

If deploying API separately, update `app/main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Deployment Checklist

- [ ] Repository pushed to GitHub
- [ ] Streamlit Cloud app created
- [ ] Main file path set to `dashboard/app_advanced.py`
- [ ] API_URL secret configured
- [ ] FastAPI backend deployed (optional)
- [ ] CORS enabled on API (if separate deployment)
- [ ] Test dashboard functionality
- [ ] Verify all visualizations work
- [ ] Check API connectivity

## Quick Deploy Commands

### For Local Testing with ngrok:

```bash
# Terminal 1: Start FastAPI
cd F:\FinSafe-Ai
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start ngrok
ngrok http 8000

# Copy the ngrok URL and add to Streamlit secrets
```

### For Production Deployment:

1. Deploy API to Railway/Render/Heroku
2. Get the public API URL
3. Add to Streamlit secrets
4. Your dashboard will automatically update!

## Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify API is accessible
3. Check browser console for errors
4. Review API logs

---

**Your dashboard will be live at:** `https://your-app-name.streamlit.app`

Happy deploying! 🚀

