"""Streamlit chat UI for the AutoStream agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

st.set_page_config(
    page_title="AutoStream Assistant",
    page_icon="🎬",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def _get_agent():
    from agent import build_agent
    return build_agent()


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []  # list of (role, content)
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {}


def _reset_session() -> None:
    st.cache_resource.clear()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.agent_state = {}


def _sidebar() -> None:
    with st.sidebar:
        st.header("AutoStream")
        st.caption("LangGraph + Gemini 3.1 Flash preview")

        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is not set. Add it to your .env file and restart.")

        st.divider()
        st.subheader("Conversation state")
        state = st.session_state.get("agent_state") or {}
        intent = state.get("intent") or "–"
        st.markdown(f"**Intent:** `{intent}`")
        st.markdown(f"**Lead mode:** `{bool(state.get('lead_mode'))}`")
        st.markdown(f"**Lead captured:** `{bool(state.get('lead_captured'))}`")

        with st.expander("Lead fields collected", expanded=True):
            st.write({
                "name": state.get("lead_name"),
                "email": state.get("lead_email"),
                "platform": state.get("lead_platform"),
            })

        if state.get("rag_context"):
            with st.expander("Last retrieved RAG context"):
                st.code(state["rag_context"], language="markdown")

        st.divider()
        if st.button("🔄 New conversation", use_container_width=True):
            _reset_session()
            st.rerun()

        st.divider()
        st.subheader("Try these")
        samples = [
            "Hi there!",
            "Tell me about your pricing.",
            "What's the difference between Basic and Pro?",
            "I want to try the Pro plan for my YouTube channel.",
        ]
        for s in samples:
            if st.button(s, use_container_width=True, key=f"sample-{s}"):
                st.session_state._pending_input = s
                st.rerun()


def _safe_md(text: str) -> str:
    """Escape characters that Streamlit's markdown renderer interprets as
    LaTeX math delimiters. Without this, a message like
    "$29/month ... $79/month" gets swallowed into a single math span and the
    whitespace/dollar signs collapse on screen."""
    if not isinstance(text, str):
        text = str(text)
    return text.replace("\\", "\\\\").replace("$", r"\$")


def _render_history() -> None:
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(_safe_md(content))


def _submit(user_text: str) -> None:
    agent = _get_agent()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.session_state.history.append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(_safe_md(user_text))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            try:
                result = agent.invoke(
                    {"messages": [HumanMessage(content=user_text)]},
                    config=config,
                )
            except Exception as exc:
                err = f"⚠️ Agent error: `{exc}`"
                placeholder.error(err)
                st.session_state.history.append(("assistant", err))
                return

        last = result["messages"][-1]
        reply = last.content if isinstance(last, AIMessage) else str(last)
        placeholder.markdown(_safe_md(reply))

        st.session_state.history.append(("assistant", reply))
        st.session_state.agent_state = {
            "intent": result.get("intent"),
            "lead_mode": result.get("lead_mode"),
            "lead_captured": result.get("lead_captured"),
            "lead_name": result.get("lead_name"),
            "lead_email": result.get("lead_email"),
            "lead_platform": result.get("lead_platform"),
            "rag_context": result.get("rag_context"),
        }


def main() -> None:
    _init_session()
    _sidebar()

    st.title("🎬 AutoStream Assistant")
    st.caption(
        "Ask about plans, pricing, or features — or tell me you're ready to sign up "
        "and I'll get your details over to the team."
    )

    _render_history()

    pending = st.session_state.pop("_pending_input", None)
    if pending:
        _submit(pending)
        st.rerun()

    user_text = st.chat_input("Type your message…")
    if user_text:
        _submit(user_text)
        st.rerun()


if __name__ == "__main__":
    main()
