# AutoStream AI Agent — Social-to-Lead Agentic Workflow

> **Built for:** ServiceHive — ML Intern Assignment  
> **Stack:** Python · LangGraph · LangChain · Local RAG · Tool Execution  
> **LLMs Supported:** Claude 3 Haiku · GPT-4o-mini · Gemini 1.5 Flash

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [How to Run Locally](#how-to-run-locally)
4. [Switching Between LLMs](#switching-between-llms)
5. [Architecture Explanation](#architecture-explanation)
6. [WhatsApp Deployment via Webhooks](#whatsapp-deployment-via-webhooks)
7. [Sample Conversation](#sample-conversation)
8. [Evaluation Checklist](#evaluation-checklist)

---

## Project Overview

This is a **Conversational AI Agent** for **AutoStream** — a fictional SaaS product that provides automated video editing tools for content creators. The agent is built on the **ServiceHive** agentic framework concept and can:

- Classify user **intent** into three categories (greeting / product inquiry / high-intent lead)
- Answer product questions using **RAG** over a local JSON knowledge base
- Detect high-intent users and **collect lead information** step by step (name → email → platform)
- Fire the **`mock_lead_capture()`** tool only after all three fields are confirmed

The agent supports **three LLMs out of the box** — you switch providers by changing a single line in `.env`.

---

## Folder Structure

```
autostream_agent/
│
├── agent.py              ← Main agent: LangGraph nodes, routing, CLI runner
├── llm_loader.py         ← Multi-LLM loader (Claude / OpenAI / Gemini)
├── knowledge_base.json   ← Local RAG source (pricing, features, policies)
├── requirements.txt      ← All Python dependencies (all 3 providers included)
├── .env.example          ← Template — copy to .env and fill in your keys
└── README.md             ← This file
```

---

## How to Run Locally

### Prerequisites
- Python **3.9 or higher**
- API key for at least **one** of the supported LLM providers

### Step 1 — Clone / Download the project
```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Create your `.env` file
```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
# Choose: claude | openai | gemini
LLM_PROVIDER=claude

ANTHROPIC_API_KEY=sk-ant-...       # Claude 3 Haiku
OPENAI_API_KEY=sk-...              # GPT-4o-mini
GOOGLE_API_KEY=AIza...             # Gemini 1.5 Flash
```

You only need to fill in the key for the provider you set in `LLM_PROVIDER`.

### Step 5 — Add `.env` to `.gitignore` (important!)
```bash
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "venv/" >> .gitignore
```

### Step 6 — Run the agent
```bash
python agent.py
```

---

## Switching Between LLMs

Open your `.env` file and change `LLM_PROVIDER` to one of the three values:

| `LLM_PROVIDER` | Model Used | API Key Needed |
|---|---|---|
| `claude` | Claude 3 Haiku (Anthropic) | `ANTHROPIC_API_KEY` |
| `openai` | GPT-4o-mini (OpenAI) | `OPENAI_API_KEY` |
| `gemini` | Gemini 1.5 Flash (Google) | `GOOGLE_API_KEY` |

**No code changes needed.** Only the `.env` value changes.

### Where to get your API keys

| Provider | URL |
|---|---|
| Anthropic (Claude) | https://console.anthropic.com |
| OpenAI (GPT) | https://platform.openai.com/api-keys |
| Google (Gemini) | https://aistudio.google.com/app/apikey |

---

## Architecture Explanation

### Why LangGraph?

LangGraph was chosen because it provides **explicit, auditable state management** through a typed `StateGraph`. Every conversation turn is a clean graph traversal: the full state enters, gets transformed by exactly one node, and exits with the updated state. This is ideal for a lead-capture workflow where routing decisions depend on accumulated state (e.g., `collecting_lead`, `lead_captured`) rather than just the current message. AutoGen's actor model suits multi-agent collaboration better; LangGraph's directed graph is simpler and more debuggable for this single-pipeline use case.

### Graph Structure

```
User Message
     │
     ▼
┌────────────┐    casual_greeting   ┌──────────────┐
│  classify  │ ──────────────────▶ │    greet     │ ──▶ END
│   (node)   │    product_inquiry  ├──────────────┤
│            │ ──────────────────▶ │     rag      │ ──▶ END
│            │    high_intent      ├──────────────┤
│            │ ──────────────────▶ │     lead     │ ──▶ END
└────────────┘  (or collecting)    └──────────────┘
```

The `classify` node runs on **every turn** and the conditional edge picks one of three action nodes. The `collecting_lead` flag short-circuits routing — if mid-collection, the agent always goes to `lead` regardless of classified intent (so a user saying "John" isn't misclassified as a greeting).

### How State Is Managed

State is a plain `TypedDict` passed **in and out of every `graph.invoke()` call** in the CLI loop:

```python
state = graph.invoke(state)   # full state persisted across turns
```

| Field | Purpose |
|---|---|
| `messages` | Full conversation history — entire list fed to LLM (last N turns) |
| `intent` | Classified label for the current turn |
| `lead_info` | Dict accumulating `name`, `email`, `platform` |
| `collecting_lead` | `True` while mid-collection — overrides intent routing |
| `lead_captured` | `True` after `mock_lead_capture()` fires |

This retains full memory across 5–6+ conversation turns with no external database.

### Multi-LLM Design

All LLM calls go through `llm_loader.py`. The `build_graph()` function accepts any `BaseChatModel` object, wrapping each node as a closure:

```python
builder.add_node("classify", lambda s: node_classify_intent(s, llm))
```

This means the graph is completely **LLM-agnostic** — swapping providers requires zero changes to the agent logic.

### RAG Pipeline

The local knowledge base (`knowledge_base.json`) is loaded at startup. `retrieve_context()` performs **keyword-based retrieval**, matching pricing / policy / product keywords to return only relevant sections as a formatted string injected into the LLM system prompt. For a larger knowledge base, this would be replaced with FAISS + embeddings for semantic search.

### Tool Execution Guard

`mock_lead_capture()` lives inside `node_lead_handler` and is called **only when the `missing` list is empty** — meaning all three fields (`name`, `email`, `platform`) are confirmed. The tool is never triggered prematurely.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business Cloud API (Meta)** with an HTTPS webhook server. Here is the complete integration design:

### Architecture

```
WhatsApp User
     │
     │  sends message
     ▼
Meta WhatsApp Cloud API
     │
     │  POST /webhook  (on every incoming message)
     ▼
Your Web Server (FastAPI)
     │
     ├── GET /webhook   ← one-time verification by Meta
     │
     └── POST /webhook  ← runs on every user message
              │
              ├─ Load session state from Redis  (key = phone number)
              ├─ graph.invoke(state)
              ├─ Save updated state to Redis    (TTL = 2 hours)
              └─ POST reply to Meta Send Message API
```

### FastAPI Webhook Server Example

```python
from fastapi import FastAPI, Request
import httpx, redis, json, os
from agent import build_graph, load_llm

app    = FastAPI()
r      = redis.Redis(host="localhost", port=6379, decode_responses=True)
graph  = build_graph(load_llm())

VERIFY_TOKEN = os.getenv("WA_VERIFY_TOKEN")
WA_TOKEN     = os.getenv("WHATSAPP_TOKEN")
PHONE_ID     = os.getenv("PHONE_NUMBER_ID")

def default_state():
    return {"messages": [], "intent": "", "lead_info": {},
            "collecting_lead": False, "lead_captured": False, "response": ""}

@app.get("/webhook")
async def verify(hub_mode: str, hub_challenge: str, hub_verify_token: str):
    if hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return {"error": "invalid token"}, 403

@app.post("/webhook")
async def receive_message(request: Request):
    body    = await request.json()
    msg     = body["entry"][0]["changes"][0]["value"]["messages"][0]
    phone   = msg["from"]
    text    = msg["text"]["body"]

    # Load or init session state from Redis
    raw     = r.get(f"session:{phone}")
    state   = json.loads(raw) if raw else default_state()

    # Run the agent for this turn
    state["messages"].append({"role": "user", "content": text})
    state = graph.invoke(state)

    # Save updated state back to Redis (2-hour TTL)
    r.setex(f"session:{phone}", 7200, json.dumps(state))

    # Send reply via WhatsApp Cloud API
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v18.0/{PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WA_TOKEN}",
                     "Content-Type": "application/json"},
            json={"messaging_product": "whatsapp",
                  "to": phone,
                  "text": {"body": state["response"]}}
        )
    return {"status": "ok"}
```

### Deployment Steps

1. **Create a Meta Developer App** at [developers.facebook.com](https://developers.facebook.com) → Add WhatsApp product → get `WHATSAPP_TOKEN` and `PHONE_NUMBER_ID`
2. **Expose your server** with HTTPS — use [ngrok](https://ngrok.com) for local testing (`ngrok http 8000`) or deploy to Render / Railway / EC2 for production
3. **Register the webhook URL** in Meta Dashboard → set callback URL to `https://your-domain.com/webhook` and set the verify token
4. **Subscribe** to the `messages` field

### Key Considerations

- **Per-user session** stored in Redis, keyed by phone number (TTL resets on each message)
- **Webhook security** — always validate the `X-Hub-Signature-256` header from Meta
- **Timeout** — WhatsApp requires a 200 OK within 15 seconds; run LLM calls async or use a background task queue
- **Rate limits** — Meta allows 80 messages/second per phone number ID

---

## Sample Conversation

```
You: Hi there!
AutoStream: Hey! Welcome to AutoStream 🎬 I'm here to help you with pricing,
            features, and getting started. What can I help you with today?

You: What's the difference between Basic and Pro?
AutoStream: Great question! Here's a quick breakdown:

            Basic ($29/month): 10 videos/month, 720p resolution, email support.
            Pro ($79/month): Unlimited videos, 4K resolution, AI captions,
            and 24/7 priority support.

You: That sounds great. I want to sign up for the Pro plan for my YouTube channel.
AutoStream: Awesome choice! 🎉 Let me get you set up. Could you please share your name?

You: Rahul Sharma
AutoStream: Thanks, Rahul! What email address should we use to reach you?

You: rahul@example.com
AutoStream: Perfect! Which creator platform do you mainly use?
            (e.g., YouTube, Instagram, TikTok)

You: YouTube
══════════════════════════════════════════════════════
  ✅  Lead captured successfully!
      Name     : Rahul Sharma
      Email    : rahul@example.com
      Platform : YouTube
══════════════════════════════════════════════════════

AutoStream: 🎉 You're all set, Rahul! We've captured your details and our team
            will reach out at rahul@example.com shortly. Welcome to AutoStream —
            excited to see your work on YouTube! 🚀
```

---

## Evaluation Checklist

| Criterion | Implementation |
|---|---|
| Agent reasoning & intent detection | `node_classify_intent` — LLM outputs one of 3 labels per turn |
| Correct RAG use | `retrieve_context()` → KB-grounded system prompt in `node_rag_respond` |
| Clean state management | `AgentState` TypedDict, fully persisted via `graph.invoke()` loop |
| Proper tool calling | `mock_lead_capture()` called only when all 3 fields confirmed |
| Code clarity & structure | Modular nodes, type hints, clear comments throughout |
| Multi-LLM support | `llm_loader.py` — switch Claude / OpenAI / Gemini via `.env` only |
| Real-world deployability | WhatsApp webhook design with Redis session store documented |
