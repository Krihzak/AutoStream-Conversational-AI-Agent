"""AutoStream conversational agent built on LangGraph + Gemini 3.1 Flash Lite.

Graph shape
-----------
                    ┌──────────────┐
    user turn ─────▶│  classify    │──────────┐
                    └──────┬───────┘          │
                           │                  │
          ┌────────────────┼──────────────────┤
          ▼                ▼                  ▼
      ┌───────┐      ┌──────────┐      ┌────────────┐
      │ greet │      │ rag_answer│     │  qualify   │
      └───┬───┘      └─────┬─────┘     └─────┬──────┘
          │                │                 │
          └────────────────┴──────── END ────┘

State is persisted via LangGraph's `MemorySaver` checkpointer keyed by a
thread id, so 5+ turns of context are retained automatically.
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from rag import KnowledgeBase
from tools import mock_lead_capture, validate_email

load_dotenv()

Intent = Literal["greeting", "product_inquiry", "high_intent", "other"]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Intent
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_mode: bool
    lead_captured: bool
    rag_context: str


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        temperature=0.2,
        max_output_tokens=512,
    )


CLASSIFIER_SYSTEM = """You are an intent classifier for AutoStream, a SaaS video-editing product.

Classify the LATEST user message into exactly one label:

- greeting           : pure social opener ("hi", "hello", "how are you", small talk)
- product_inquiry    : question about pricing, plans, features, policies, or the product itself
- high_intent        : user signals they want to sign up, try, buy, subscribe, get started, or wants the team to contact them. Phrases like "I want to try", "sign me up", "ready to buy", "interested in the Pro plan for my channel" count as high intent.
- other              : anything else (off-topic, unclear)

Return ONLY a compact JSON object: {"intent": "<label>"}."""


LEAD_EXTRACTOR_SYSTEM = """Extract lead fields from the user's latest message.

Return ONLY JSON with these keys (use null for anything not clearly present):
{"name": <string|null>, "email": <string|null>, "platform": <string|null>}

Rules:
- Only extract a NAME if the user is clearly offering their own name.
- Only extract an EMAIL if a valid email address is present.
- PLATFORM is a creator platform: YouTube, Instagram, TikTok, Twitch, LinkedIn, Facebook, X/Twitter, etc. Normalize casing (e.g. "youtube" -> "YouTube").
- Do not invent values. Prefer null over guessing."""


ANSWER_SYSTEM = """You are AutoStream's helpful sales & support assistant.

Answer the user's question using ONLY the knowledge base context below. If the
context does not contain the answer, say you don't have that information and
offer to connect them with the team. Keep replies concise (2-4 sentences).

Knowledge base context:
{context}"""


kb = KnowledgeBase()


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _last_user_text(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def classify_node(state: AgentState) -> dict:
    """Classify intent. If we're already mid-lead-capture, stay in that mode
    regardless of what the user types — premature exits are the #1 cause of
    broken multi-turn flows."""
    if state.get("lead_mode") and not state.get("lead_captured"):
        return {"intent": "high_intent"}

    user_text = _last_user_text(state)
    llm = _llm()
    reply = llm.invoke(
        [SystemMessage(content=CLASSIFIER_SYSTEM), HumanMessage(content=user_text)]
    )
    parsed = _parse_json(reply.content if isinstance(reply.content, str) else "")
    intent: Intent = parsed.get("intent", "other")
    if intent not in ("greeting", "product_inquiry", "high_intent", "other"):
        intent = "other"
    return {"intent": intent}


def greet_node(state: AgentState) -> dict:
    user_text = _last_user_text(state)
    llm = _llm()
    reply = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are AutoStream's friendly assistant. Greet the user warmly in "
                    "1-2 sentences and invite them to ask about pricing, features, or "
                    "getting started."
                )
            ),
            HumanMessage(content=user_text),
        ]
    )
    return {"messages": [AIMessage(content=reply.content)]}


def rag_answer_node(state: AgentState) -> dict:
    user_text = _last_user_text(state)
    docs = kb.retrieve(user_text, k=3)
    context = kb.format_context(docs)

    history = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))]
    llm = _llm()
    reply = llm.invoke(
        [SystemMessage(content=ANSWER_SYSTEM.format(context=context)), *history]
    )
    return {
        "messages": [AIMessage(content=reply.content)],
        "rag_context": context,
    }


def qualify_node(state: AgentState) -> dict:
    """Collects name, email, platform. Only calls the tool once all three are
    present and the email is well-formed."""
    user_text = _last_user_text(state)
    llm = _llm()

    extraction_reply = llm.invoke(
        [
            SystemMessage(content=LEAD_EXTRACTOR_SYSTEM),
            HumanMessage(content=user_text),
        ]
    )
    extracted = _parse_json(
        extraction_reply.content if isinstance(extraction_reply.content, str) else ""
    )

    name = state.get("lead_name") or (extracted.get("name") or None)
    email = state.get("lead_email") or (extracted.get("email") or None)
    platform = state.get("lead_platform") or (extracted.get("platform") or None)

    if email and not validate_email(email):
        email = None

    missing: List[str] = []
    if not name:
        missing.append("your full name")
    if not email:
        missing.append("your email address")
    if not platform:
        missing.append("the platform you create for (e.g. YouTube, Instagram)")

    updates: dict = {
        "lead_name": name,
        "lead_email": email,
        "lead_platform": platform,
        "lead_mode": True,
    }

    if missing:
        have_parts = []
        if name:
            have_parts.append(f"name: {name}")
        if email:
            have_parts.append(f"email: {email}")
        if platform:
            have_parts.append(f"platform: {platform}")
        have_str = f" I've got {', '.join(have_parts)}." if have_parts else ""
        ask = " and ".join(missing)
        reply_text = (
            f"Great — I'd love to get you set up.{have_str} "
            f"Could you share {ask}?"
        )
        updates["messages"] = [AIMessage(content=reply_text)]
        return updates

    confirmation = mock_lead_capture(name, email, platform)
    updates["lead_captured"] = True
    updates["lead_mode"] = False
    updates["messages"] = [
        AIMessage(
            content=(
                f"Perfect, thanks {name}! You're all set — our team will reach out to "
                f"{email} shortly with next steps for your {platform} workflow. "
                f"({confirmation})"
            )
        )
    ]
    return updates


def fallback_node(state: AgentState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm AutoStream's assistant — I can help with pricing, plan "
                    "features, policies, or getting you signed up. What would you "
                    "like to know?"
                )
            )
        ]
    }


def route_after_classify(state: AgentState) -> str:
    intent = state.get("intent", "other")
    if intent == "greeting":
        return "greet"
    if intent == "product_inquiry":
        return "rag_answer"
    if intent == "high_intent":
        return "qualify"
    return "fallback"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("greet", greet_node)
    graph.add_node("rag_answer", rag_answer_node)
    graph.add_node("qualify", qualify_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "greet": "greet",
            "rag_answer": "rag_answer",
            "qualify": "qualify",
            "fallback": "fallback",
        },
    )
    graph.add_edge("greet", END)
    graph.add_edge("rag_answer", END)
    graph.add_edge("qualify", END)
    graph.add_edge("fallback", END)

    return graph.compile(checkpointer=MemorySaver())
