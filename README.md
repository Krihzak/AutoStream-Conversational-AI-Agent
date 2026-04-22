# AutoStream Conversational AI Agent

A multi-turn conversational agent for **AutoStream**, a fictional SaaS that offers automated video editing for content creators. The agent classifies intent, answers product/pricing questions from a local knowledge base using RAG, and captures qualified leads via a tool call — all while maintaining conversation state across turns.

Built with **LangGraph + Gemini 3.1 Flash Lite**, and shipped with a **Streamlit chat UI** that exposes the agent's live state so you can see intent, lead fields, and retrieved RAG context update turn-by-turn.

### At a glance
- 🧠 **Intent detection** — greeting / product inquiry / high-intent lead
- 📚 **RAG** — local JSON knowledge base + TF-IDF retriever, no vector DB required
- 🛠️ **Tool calling** — `mock_lead_capture(name, email, platform)` fires only when all three fields are present *and* the email is valid
- 💾 **Memory** — LangGraph `MemorySaver` checkpointer keyed by `thread_id`, state persists across 5–6+ turns
- 🖥️ **UI** — Streamlit chat app with a live debug sidebar (intent, lead state, RAG context)

---

## 1. How to run locally

### Prerequisites
- Python 3.9+
- A Google AI Studio API key — get one at https://aistudio.google.com/app/apikey
- A current Gemini Flash-class model. The project defaults to **`gemini-3.1-flash-lite`** — the newest, cheapest, lowest-latency option in the Gemini 3 family. The spec originally called for Gemini 1.5 Flash, which Google has since retired. Override via `GEMINI_MODEL` in `.env` if your key/region prefers `gemini-3.1-flash`, `gemini-2.5-flash-lite`, or `gemini-2.5-flash`.

### Setup
```bash
# 1. Clone and enter the project
cd ServiceHive

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY
```

Your `.env` should look like:
```env
GOOGLE_API_KEY=AIza...your-actual-key...
GEMINI_MODEL=gemini-3.1-flash-lite
```

### Run

The project ships with two entry points — pick whichever fits your workflow:

| Entry point | Command | When to use |
|---|---|---|
| **Streamlit web UI** | `streamlit run app.py` | Best for exploring the agent & recording the demo video |
| **Interactive CLI** | `python main.py` | Fastest terminal-only smoke test |


#### Web UI (recommended)
```bash
streamlit run app.py
```
Opens at <http://localhost:8501>. What you get:
- A `st.chat_message` / `st.chat_input` conversation that persists across Streamlit reruns.
- A **live debug sidebar** showing the current classified `intent`, the `lead_mode` latch, the `lead_captured` flag, every lead field collected so far, and the most recent RAG context passage that was retrieved. Great for demonstrating that the state machine and RAG retrieval are actually doing what the code claims.
- Four **quick-prompt buttons** that walk through the greeting → pricing → feature compare → high-intent sign-up flow in one click each.
- A **🔄 New conversation** button that rotates the LangGraph `thread_id` so you can reset memory without restarting the server.

#### CLI
```bash
python main.py
```

### Example session
```
You: Hi, tell me about your pricing.
Agent: AutoStream offers two plans. Basic is $29/month (10 videos, 720p).
       Pro is $79/month (unlimited videos, 4K, AI captions, 24/7 support).

You: Sounds good — I want to try the Pro plan for my YouTube channel.
Agent: Great — I'd love to get you set up. Could you share your full name and
       your email address?

You: I'm Alexa, alexa1234@gmail.com
Agent: Perfect, thanks Aman! You're all set — our team will reach out shortly.
       (Lead captured successfully: Alexa, alexa1234@gmail.com, YouTube)
```

### Demo-video script (copy-paste into the Streamlit UI)
Use this 5-turn script to showcase every evaluation criterion in under 90 seconds:

1. `Hi there!` — shows greeting intent
2. `Tell me about your pricing.` — shows RAG retrieval + grounded answer
3. `What's the difference between Basic and Pro?` — shows the agent retains context about *which* product is in play
4. `I want to try the Pro plan for my YouTube channel.` — shows intent shift to `high_intent`; platform auto-extracted
5. `I'm Alexa, alexa1234@gmail.com` — shows tool call firing only once all three fields are present

On turn 5 you should see the sidebar flip `lead_captured: True` and the terminal running Streamlit will print:
```
Lead captured successfully:Alexa, Alexa1234, YouTube
```

### Verifying memory across 5–6 turns
Three ways to prove state persists:

1. **Sidebar tells the story.** Every turn updates the `intent`, `lead_mode`, `lead_captured`, and lead-fields panels. Values collected on earlier turns (e.g. `platform=YouTube` from turn 4) are still there on turn 5 — that's the memory.
2. **Tool arguments.** The final `mock_lead_capture()` call uses name from one turn, platform from another, and email from yet another. Only possible if the checkpointer persisted state between turns.
3. **Negative control.** Click **🔄 New conversation**. The sidebar resets, and the same questions get treated as if you just arrived — proving the memory is real and `thread_id`-scoped, not accidental statelessness.

---

## 2. Architecture 

**Stack:** LangGraph for orchestration, **Gemini 3.1 Flash Lite** (via `langchain-google-genai`) as the LLM, and a local JSON knowledge base with a built-in TF-IDF retriever for RAG.

**Why LangGraph?** The problem is a *stateful state machine*, not a single LLM call. Intent classification routes to distinct behaviours (greet / retrieve / qualify), and the lead-capture branch has to accumulate fields across multiple turns before firing the tool. LangGraph models this natively as a directed graph with typed state, conditional edges, and a built-in checkpointer — far cleaner than wiring the same control flow inside an AutoGen multi-agent chat. LangGraph also makes it trivial to visualize and reason about every path a conversation can take, which matters for production readiness.

**Graph shape.** `classify → {greet | rag_answer | qualify | fallback} → END`. The classifier node calls Gemini 3.1 Flash Lite with a strict JSON schema to label each turn. `rag_answer` retrieves from a local JSON knowledge base via a lightweight TF-IDF retriever (`rag.py`) and grounds generation in the retrieved passages. `qualify` extracts lead fields from free-form text, persists them, asks for whatever's missing, and only calls `mock_lead_capture()` once *name, valid email, and platform* are all present — preventing premature tool invocation.

**State management.** An `AgentState` TypedDict holds messages, current intent, partial lead fields, a `lead_mode` latch, and the last RAG context. A `MemorySaver` checkpointer keyed by `thread_id` persists state across turns, giving the agent memory for 5–6+ turns out of the box. Once the user enters lead-capture mode, the classifier is short-circuited so intent drift (e.g. the user asking a tangential question) doesn't abort the collection flow.

---

## 3. WhatsApp Deployment (Webhooks)

To put this agent on WhatsApp, wrap it behind a webhook endpoint and connect the **Meta WhatsApp Business Cloud API** (or Twilio's WhatsApp API).

**Flow:**
1. **Register a WhatsApp Business number** in Meta's Business Manager and enable the Cloud API. Meta provides a Phone Number ID and a permanent access token.
2. **Expose a webhook** — e.g. `POST /whatsapp/webhook` on a FastAPI/Flask app deployed to Render, Fly.io, or AWS Lambda behind API Gateway. Register its public HTTPS URL in the WhatsApp app's webhook config, along with a verification token (Meta sends a `GET` challenge first — respond with the `hub.challenge` value).
3. **Handle inbound messages.** When a user texts the number, Meta POSTs a JSON payload (`entry[].changes[].value.messages[]`) to the webhook. Extract the `from` (user's phone number, used as `thread_id`) and `text.body`.
4. **Invoke the agent** — call `agent.invoke({"messages": [HumanMessage(content=text)]}, config={"configurable": {"thread_id": from_number}})`. LangGraph's checkpointer keeps state per user automatically.
5. **Reply** via `POST https://graph.facebook.com/v20.0/<PHONE_NUMBER_ID>/messages` with `{"messaging_product": "whatsapp", "to": from_number, "text": {"body": agent_reply}}` and the access token as a bearer header.
6. **Production hardening:** verify the `X-Hub-Signature-256` HMAC on every request, enqueue work to a background worker (Celery/RQ) so the webhook returns `200` in under 5 seconds (Meta retries otherwise), swap `MemorySaver` for a durable checkpointer (Postgres/Redis) so state survives restarts, and add rate limiting plus structured logging for auditability.

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `404 models/<name> is not found for API version v1beta` | Model ID not available for your key/region (Gemini 1.5 Flash was retired). | Edit `.env`: try `GEMINI_MODEL=gemini-3.1-flash-lite` → `gemini-3.1-flash` → `gemini-2.5-flash-lite` → `gemini-2.5-flash`, then restart. |
| `GOOGLE_API_KEY is not set` | `.env` missing or not loaded from the run directory. | Ensure `.env` sits next to `app.py` / `main.py`; restart the process (env is loaded at startup via `python-dotenv`). |
| Sidebar shows `lead_captured: False` after the capture turn | Streamlit render-order race (the sidebar renders before the submit handler writes state). | Already patched in `app.py` via `st.rerun()` after every submit — hard-refresh the page (`Ctrl+Shift+R`) if you're on a stale cached version. |
| Agent seems to "forget" on every turn | New `thread_id` each turn, or the Streamlit process was restarted (`MemorySaver` is in-memory only). | Stay in one browser session; don't click *New conversation* mid-flow. For persistence across restarts, swap `MemorySaver` → `SqliteSaver` in `agent.py::build_agent`. |
| Tool fired prematurely / with missing fields | Extractor hallucinated a field. | The code already validates the email regex and short-circuits on `null`; if it still happens, raise the temperature back down to `0` in `agent.py::_llm`. |

---

## 5. Project Layout
```
ServiceHive/
├── agent.py              # LangGraph state machine (classify / greet / rag / qualify)
├── rag.py                # Local knowledge base + TF-IDF retriever
├── tools.py              # mock_lead_capture + email validation
├── app.py                # Streamlit web UI
├── main.py               # Interactive CLI
├── knowledge_base.json   # Pricing, features, policies (RAG source)
├── requirements.txt
├── .env.example
└── README.md
```

---

| Criterion | Where it lives |
|---|---|
| Agent reasoning & intent detection | `agent.py::classify_node` + `CLASSIFIER_SYSTEM` (Gemini 3.1 Flash Lite, JSON-schema output) |
| Correct use of RAG | `rag.py` (retrieval) + `agent.py::rag_answer_node` (grounded generation via Gemini 3.1 Flash Lite) |
| Clean state management | `AgentState` TypedDict + LangGraph `MemorySaver` checkpointer |
| Proper tool calling logic | `agent.py::qualify_node` — gated on all three fields + valid email |
| Code clarity & structure | Modules split by concern (`rag.py` / `tools.py` / `agent.py` / `app.py` / `main.py`) |
| Real-world deployability | `.env` config, pinned deps, Streamlit UI for end users, stateless graph + durable-checkpointer swap, WhatsApp webhook plan above |

---
