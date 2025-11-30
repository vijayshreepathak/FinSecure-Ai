# ✅ Streamlit Cloud Deployment Checklist

## Quick Deploy (5 Minutes)

### Step 1: Go to Streamlit Cloud
👉 [https://share.streamlit.io](https://share.streamlit.io)

### Step 2: Sign In
- Click "Sign in"
- Authorize with GitHub

### Step 3: Deploy Your App
1. Click **"New app"** button
2. Fill in:
   - **Repository**: `vijayshreepathak/FinSecure-Ai`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app_advanced.py`
3. Click **"Deploy!"**

### Step 4: Configure API URL
1. Wait for deployment (1-2 minutes)
2. Click **⋮** (three dots) → **Settings**
3. Go to **Secrets** tab
4. Add:
   ```toml
   API_URL = "http://localhost:8000"
   ```
   *(Update this after deploying your API)*

### Step 5: Test Your App
- Your app will be live at: `https://your-app-name.streamlit.app`
- Test the dashboard functionality

## 🚀 Deploy FastAPI Backend (Optional)

### Option 1: Railway (Recommended)
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select your repository
4. Set root directory: `app/`
5. Add start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Copy the Railway URL
7. Update Streamlit secrets with Railway URL

### Option 2: Render
1. Go to [render.com](https://render.com)
2. New Web Service
3. Connect GitHub
4. Build: `pip install -r requirements.txt`
5. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Copy Render URL
7. Update Streamlit secrets

### Option 3: ngrok (For Testing)
```bash
# Terminal 1: Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Start ngrok
ngrok http 8000

# Copy ngrok URL to Streamlit secrets
```

## ✅ Final Checklist

- [ ] Streamlit app deployed
- [ ] API_URL secret configured
- [ ] FastAPI backend deployed (or using ngrok)
- [ ] Dashboard loads successfully
- [ ] Can connect to API
- [ ] Predictions work
- [ ] Visualizations display correctly

## 🎉 You're Done!

Your dashboard is now live on Streamlit Cloud! 🚀

---

**Need help?** Check [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for detailed instructions.

