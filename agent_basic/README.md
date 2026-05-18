# aiseasy

A simple, extendable local AI agent project for learning. This version uses a publicly available Hugging Face model and does not require an API key or external access credentials.

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Run the agent:

```bash
python3 main.py
```

3. Open the notebook for guided learning:

- `simple_ai_agent_demo.ipynb`

## Usage

- Ask a question such as: `Tell me a short story about a robot.`
- Try commands like `help` to see usage tips.
- Type `exit`, `quit`, or `q` to stop.

## Project structure

- `main.py` – CLI entrypoint for the agent.
- `ai_agent.py` – simple local AI agent implementation using a public model.
- `requirements.txt` – Python dependencies.
- `simple_ai_agent_demo.ipynb` – notebook that introduces the project and runs the agent.
- `.gitignore` – ignores local environment and Python cache files.

## Extendability

The agent is designed to be easy to extend:
- add new functions in `ai_agent.py`
- expand `SimpleAIAgent.handle_user_request`
- add notebook examples in `simple_ai_agent_demo.ipynb`
