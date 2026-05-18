# ReAct Agent Project

This dedicated learning project shows a ReAct-style AI agent using the OpenAI API.

## What is included

- `react_agent.py` — a ReAct agent harness with:
  - OpenAI chat completion integration
  - a simple MCP-style context manager
  - memory management for user and tool interactions
  - prompt engineering for system, memory, and tool channels
- `main.py` — CLI launcher for the agent
- `react_agent_learning.ipynb` — notebook with clear explanation and step-by-step examples
- `requirements.txt` — dependencies for OpenAI and environment loading
- `.env.example` — API key setup guidance

## Learning goals

- Understand the ReAct pattern
- Explore how an agent harness orchestrates prompt construction, tools, and memory
- Learn about memory management, context engineering, and MCP-style channels
- Start with one pattern and extend it later with Plan-and-Execute or Reflection

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and add your OpenAI API key.

3. Run the agent:

```bash
python3 main.py
```

4. Open the notebook `react_agent_learning.ipynb` in VS Code for guided learning.
