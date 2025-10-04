# LangGraph Simple Agent

This folder contains a minimal example of building an agent with [LangGraph](https://github.com/langchain-ai/langgraph).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure the `langgraph` package is installed. The project currently lists it in the shared `requirements.txt`.
2. (Optional) Activate the repo's virtual environment if you maintain one.

## Usage

Run the command-line demo:
```bash
python3 ai_agents/langgraph_simple_agent.py
```
Type messages to the agent and it will respond to greetings, simple addition questions, and help prompts. Use `quit` or `exit` to leave the session.

## Customisation

- Replace `simple_responder` in `langgraph_simple_agent.py` with your own logic that calls an LLM or integrates tools.
- Expand the LangGraph by adding more nodes with `graph.add_node(...)` and connecting them via edges to build richer workflows.
