"""Minimal LangGraph agent example.

This script builds a tiny stateful agent with LangGraph that can run in a CLI.
The agent recognises simple greetings, answers addition questions, and provides
fallback replies for anything else. It demonstrates how to use `StateGraph`
with a typed state and a single tool node.

Run it with:
    python ai_agents/langgraph_simple_agent.py

Dependencies:
    - langgraph
    - langchain-core (installed automatically with langgraph)

Set the `OPENAI_API_KEY` environment variable if you later swap in an LLM-based
response function. The current example is fully deterministic and requires no
network access.
"""

from __future__ import annotations

import re
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State container tracked by LangGraph."""

    messages: Annotated[List[BaseMessage], add_messages]


def simple_responder(state: AgentState) -> AgentState:
    """Respond to the most recent user message with simple pattern matching."""

    last_message = state["messages"][-1].content.strip()
    lowered = last_message.lower()

    if not last_message:
        reply = "I need a question or statement to help with."
    elif any(greet in lowered for greet in ("hello", "hi", "hey")):
        reply = "Hello! Ask me to add some numbers or say 'quit' to exit."
    elif _looks_like_addition(lowered):
        reply = _handle_addition(last_message)
    elif "help" in lowered:
        reply = (
            "I am a simple demo agent built with LangGraph."
            " Try asking: 'add 3 and 4'."
        )
    else:
        reply = (
            "I only handle greetings, addition questions, and simple help prompts."
            " Try something like 'what is the sum of 5 and 7?'."
        )

    return {"messages": [AIMessage(content=reply)]}


def _looks_like_addition(message: str) -> bool:
    """Heuristic to detect if the user is asking for addition."""

    return "add" in message or "sum" in message or bool(re.search(r"\\d+\\s*\\+\\s*\\d+", message))


def _handle_addition(message: str) -> str:
    """Extract numbers from the message and return their sum."""

    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", message)
    if not numbers:
        return "Tell me which numbers to add, e.g. 'add 2 and 5'."

    values = [float(num) for num in numbers]
    total = sum(values)

    if all(num.isdigit() for num in numbers):
        return f"The answer is {int(total)}."

    return f"The answer is {total}."


def build_agent() -> StateGraph:
    """Configure and compile the LangGraph agent."""

    graph = StateGraph(AgentState)
    graph.add_node("responder", simple_responder)
    graph.add_edge(START, "responder")
    graph.add_edge("responder", END)

    return graph.compile()


def run_cli() -> None:
    """Interactive command-line loop for the agent."""

    agent = build_agent()
    state: AgentState = {"messages": []}

    print("Simple LangGraph agent ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = agent.invoke(state)
        response = state["messages"][-1]
        print(f"Agent: {response.content}")


if __name__ == "__main__":
    run_cli()
