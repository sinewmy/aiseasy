def execute(query: str) -> str:
    """Search a built-in knowledge base for facts or definitions."""
    knowledge = {
        "react": "ReAct is a reasoning strategy where the agent alternates between reasoning and actions.",
        "mcp": "MCP here means using separate message channels for system, memory, and tool context.",
        "memory": "Memory management helps the agent remember recent conversation turns and tool outputs.",
    }
    return knowledge.get(query.strip().lower(), f"No knowledge found for '{query.strip()}'.")
