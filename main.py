"""Interactive CLI for the AutoStream agent."""

from __future__ import annotations

import os
import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from agent import build_agent


def run() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is not set. Add it to your environment or .env file.")
        sys.exit(1)

    agent = build_agent()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("AutoStream Assistant — type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        last = result["messages"][-1]
        if isinstance(last, AIMessage):
            print(f"Agent: {last.content}\n")


if __name__ == "__main__":
    run()
