import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

@dataclass
class AgentTool:
    name: str
    description: str
    function: Callable[[str], str]


@dataclass
class AgentSkill:
    name: str
    description: str
    keywords: List[str]
    tool_name: Optional[str] = None


class MemoryStore:
    """A simple memory manager for agent conversations and facts."""

    def __init__(self, capacity: int = 6) -> None:
        self.capacity = capacity
        self.history: List[Dict[str, str]] = []

    def add_interaction(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.capacity:
            self.history.pop(0)

    def get_recent_memory(self) -> str:
        if not self.history:
            return "No previous memory available."
        lines = [f"{item['role']}: {item['content']}" for item in self.history]
        return "\n".join(lines)


class MCPContext:
    """A minimal MCP-style context manager for channel separation."""

    def __init__(self) -> None:
        self.system: List[str] = []
        self.memory: List[str] = []
        self.tools: List[str] = []
        self.skills: List[str] = []

    def add_system(self, text: str) -> None:
        self.system.append(text)

    def add_memory(self, text: str) -> None:
        self.memory.append(text)

    def add_tool(self, text: str) -> None:
        self.tools.append(text)

    def add_skill(self, text: str) -> None:
        self.skills.append(text)

    def build_messages(self, user_input: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": "\n".join(self.system)})
        if self.memory:
            messages.append({"role": "system", "content": "Memory:\n" + "\n".join(self.memory)})
        if self.tools:
            messages.append({"role": "system", "content": "Tool descriptions:\n" + "\n".join(self.tools)})
        if self.skills:
            messages.append({"role": "system", "content": "AI Skills:\n" + "\n".join(self.skills)})
        messages.append({"role": "user", "content": user_input})
        return messages


class ReActAgentHarness:
    """A harness for a ReAct-style agent using OpenAI chat completions."""

    ACTION_REGEX = re.compile(r"Action:\s*(.+?)\nAction Input:\s*(.+)", re.S)

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed explicitly.")

        self.client = OpenAI(api_key=self.api_key)
        self.memory = MemoryStore(capacity=8)
        self.context = MCPContext()
        self.tools: Dict[str, AgentTool] = {}
        self.skills: Dict[str, AgentSkill] = {}
        self.last_selected_skill: Optional[AgentSkill] = None
        self._prepare_context()

    def _prepare_context(self) -> None:
        self.context.add_system(
            "You are a ReAct-style reasoning agent. Use the ReAct pattern to think, choose an action,"
            " and call tools when appropriate. If the answer is final, provide a clear response without action tags."
        )
        self.context.add_system(
            "If you need an external action, respond with:\nAction: <tool_name>\nAction Input: <input>"
        )
        self.context.add_tool(
            "search_knowledge: Search the knowledge base for facts or definitions. Input should be a short query."
        )
        self.context.add_tool(
            "calculate: Perform a basic arithmetic or logic calculation based on the input."
        )
        self.context.add_memory("The agent should keep memory of the last interactions and tool outputs.")
        self.context.add_skill(
            "knowledge: A skill for factual search and definitions. Use it when the user asks about a concept, definition, or explanation."
        )
        self.context.add_skill(
            "calculator: A skill for arithmetic and logic calculation. Use it when the user asks for math or simple data computation."
        )
        self.context.add_skill(
            "general: A fallback skill for open-ended reasoning, summaries, or conversational replies."
        )

    def register_tool(self, tool: AgentTool) -> None:
        self.tools[tool.name] = tool

    def register_skill(self, skill: AgentSkill) -> None:
        self.skills[skill.name] = skill

    def build_prompt(self, user_input: str, selected_skill: AgentSkill) -> List[Dict[str, str]]:
        messages = self.context.build_messages(user_input)
        if selected_skill is not None:
            messages.insert(
                -1,
                {
                    "role": "system",
                    "content": (
                        f"Selected skill: {selected_skill.name}\n"
                        f"{selected_skill.description}\n"
                        f"Preferred tool: {selected_skill.tool_name or 'none'}."
                    ),
                },
            )
        return messages

    def _choose_skill(self, user_input: str) -> AgentSkill:
        lowercase_input = user_input.lower()
        for skill in self.skills.values():
            if any(keyword in lowercase_input for keyword in skill.keywords):
                return skill
        return self.skills["general"]

    def _parse_action(self, text: str) -> Optional[Dict[str, str]]:
        match = self.ACTION_REGEX.search(text)
        if not match:
            return None
        return {"action": match.group(1).strip(), "input": match.group(2).strip()}

    def _run_tool(self, action_name: str, action_input: str) -> str:
        if action_name not in self.tools:
            return f"Unknown tool: {action_name}"
        tool = self.tools[action_name]
        return tool.function(action_input)

    def run(self, user_input: str, max_steps: int = 2) -> str:
        self.memory.add_interaction("user", user_input)
        selected_skill = self._choose_skill(user_input)
        self.last_selected_skill = selected_skill
        self.memory.add_interaction("assistant", f"Selected skill: {selected_skill.name}")

        response = ""
        for step in range(max_steps):
            messages = self.build_prompt(user_input if step == 0 else response, selected_skill)
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            response = completion.choices[0].message.content.strip()
            self.memory.add_interaction("assistant", response)

            action = self._parse_action(response)
            if action is None:
                return response

            tool_result = self._run_tool(action["action"], action["input"])
            self.memory.add_interaction("tool", f"{action['action']}: {tool_result}")
            user_input = (
                f"Tool output for {action['action']}: {tool_result}\n"
                "Please continue your reasoning and provide the final answer."
            )

        return response


def search_knowledge(query: str) -> str:
    knowledge = {
        "react": "ReAct is a reasoning strategy where the agent alternates between reasoning and actions.",
        "mcp": "MCP here means using separate message channels for system, memory, and tool context.",
        "memory": "Memory management helps the agent remember recent conversation turns and tool outputs.",
    }
    return knowledge.get(query.strip().lower(), f"No knowledge found for '{query.strip()}'.")


def calculate(expression: str) -> str:
    try:
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {})
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


def create_default_agent() -> ReActAgentHarness:
    agent = ReActAgentHarness()
    agent.register_tool(AgentTool("search_knowledge", "Search a built-in knowledge base.", search_knowledge))
    agent.register_tool(AgentTool("calculate", "Evaluate arithmetic expressions.", calculate))
    agent.register_skill(
        AgentSkill(
            "knowledge",
            "Use the knowledge skill when the user asks for a definition, concept, or factual explanation.",
            ["define", "what is", "who is", "explain", "describe", "meaning", "fact", "knowledge"],
            tool_name="search_knowledge",
        )
    )
    agent.register_skill(
        AgentSkill(
            "calculator",
            "Use the calculator skill when the user asks to compute a number, evaluate an expression, or solve a simple math task.",
            ["calculate", "compute", "add", "subtract", "multiply", "divide", "sum", "math", "evaluate"],
            tool_name="calculate",
        )
    )
    agent.register_skill(
        AgentSkill(
            "general",
            "Use the general skill for conversational replies, summaries, or questions that do not clearly map to a specific tool.",
            [],
            tool_name=None,
        )
    )
    return agent
