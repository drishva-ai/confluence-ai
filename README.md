# Confluence AI Chatbot

Ask questions about your Confluence documentation using AI.
Works with pages, PDFs, Word docs, and Excel files.

## Quick Start (5 minutes)

### Step 1 — Get a free Groq API key
Go to: https://console.groq.com
Click "Create API Key" → copy it

### Step 2 — Configure
```bash
cp backend/.env.example backend/.env
```
Edit `backend/.env` and fill in:
```
GROQ_API_KEY=your_key_here
CONFLUENCE_URL=https://yourcompany.atlassian.net
CONFLUENCE_USERNAME=your@email.com
CONFLUENCE_API_TOKEN=your_confluence_token
CONFLUENCE_SPACE_KEY=YOUR_SPACE
```

Get Confluence API token: https://id.atlassian.com/manage-profile/security/api-tokens

### Step 3 — Run
```bash
docker compose up --build
```

### Step 4 — Sync your Confluence
```bash
curl -X POST http://localhost:8000/api/sync
```

### Step 5 — Open the chat
Go to: http://localhost:3000

---

## Switch AI Model (ONE line change)

Edit `backend/.env`:

```bash
# Free (default)
AI_PROVIDER=groq
AI_MODEL=llama-3.1-70b-versatile

# Upgrade to GPT-4o
AI_PROVIDER=openai
AI_MODEL=gpt-4o

# Use Claude
AI_PROVIDER=anthropic
AI_MODEL=claude-3-5-sonnet-20241022

# Fully local (free, needs GPU)
AI_PROVIDER=ollama
AI_MODEL=llama3.2
```

Then: `docker compose up --build`

---

## Embed in any website

Add this one line to your HTML:
```html
<script src="https://yourserver.com/widget.js"></script>
```

Or in React:
```jsx
import ChatWidget from './ChatWidget';
// Add anywhere in your app:
<ChatWidget />
```

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| POST | /api/sync | Sync Confluence content |
| POST | /api/chat | Ask a question |
| GET | /api/chat/stream | Streaming response |
| GET | /api/stats | View sync stats |
| GET | /health | Health check |

---

## Pricing for your customers

| Plan | Price | Includes |
|------|-------|---------|
| Starter | ₹2,999/month | 1 space, 500 queries |
| Growth | ₹7,999/month | 5 spaces, 5000 queries |
| Enterprise | ₹24,999/month | Unlimited, on-premise |

Your cost to run per customer: ~₹500/month (AI API + server)
Your profit margin: ~85%
