import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from pattern import ActionResult, AgentPattern
from registry import SkillRegistry, ToolRegistry

load_dotenv()


class MemoryStore:
    def __init__(self, capacity: int = 6) -> None:
        self.capacity = capacity
        self.history: List[Dict[str, str]] = []

    def add_interaction(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.capacity:
            self.history.pop(0)

    def get_recent_memory(self) -> str:
        if not self.history:
            return ""
        return "\n".join(f"{item['role']}: {item['content']}" for item in self.history)


class PromptContext:
    def __init__(self) -> None:
        self.system: List[str] = []
        self.memory_guidance: List[str] = []

    def add_system(self, text: str) -> None:
        self.system.append(text)

    def add_memory_guidance(self, text: str) -> None:
        self.memory_guidance.append(text)

    def build_messages(self, user_input: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": "\n".join(self.system)})
        if self.memory_guidance:
            messages.append(
                {"role": "system", "content": "Memory guidance:\n" + "\n".join(self.memory_guidance)}
            )
        messages.append({"role": "user", "content": user_input})
        return messages


class BaseAgentHarness:
    def __init__(
        self,
        pattern: AgentPattern,
        api_key: Optional[str] = None,
        system_prompt_file: Optional[Path] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed explicitly.")

        self.client = OpenAI(api_key=self.api_key)
        self.pattern = pattern
        self.context = PromptContext()
        self.memory = MemoryStore(capacity=8)
        self.tool_registry = ToolRegistry()
        self.skill_registry = SkillRegistry()
        self.last_selected_skill: Optional[str] = None

        self._prepare_context(system_prompt_file)
        self.tool_registry.load_from_directory()
        self.skill_registry.load_from_directory()

    def _load_system_prompt(self, system_prompt_file: Optional[Path] = None) -> str:
        if system_prompt_file is None:
            system_prompt_file = Path(__file__).resolve().parent / "system_prompt.md"
        return system_prompt_file.read_text(encoding="utf-8").strip()

    def _prepare_context(self, system_prompt_file: Optional[Path] = None) -> None:
        self.context.add_system(self._load_system_prompt(system_prompt_file))
        if self.pattern.system_prompt_file is not None:
            self.context.add_system(self._load_system_prompt(self.pattern.system_prompt_file))
        elif self.pattern.system_instructions:
            self.context.add_system(self.pattern.system_instructions)
        self.context.add_memory_guidance("The agent should keep memory of the last interactions and tool outputs.")

    def build_prompt(self, user_input: str) -> List[Dict[str, str]]:
        base_messages = self.context.build_messages(user_input)
        history = self.memory.get_recent_memory()
        if history:
            base_messages.insert(-1, {"role": "system", "content": "Recent interactions:\n" + history})

        tool_text = self.tool_registry.get_descriptions()
        skill_text = self.skill_registry.get_descriptions()
        return self.pattern.format_prompt(base_messages, tool_text, skill_text)

    def _validate_action(self, action: ActionResult) -> Optional[str]:
        if action.skill:
            skill = self.skill_registry.get_skill(action.skill)
            if skill is None:
                return f"Unknown skill: {action.skill}"
            if action.action and skill.tool_name and skill.tool_name != action.action:
                return (
                    f"Skill '{action.skill}' prefers tool '{skill.tool_name}',"
                    f" but got '{action.action}'."
                )
        if action.action:
            tool = self.tool_registry.get_tool(action.action)
            if tool is None:
                return f"Unknown tool: {action.action}"
        return None

    def run(self, user_input: str, max_steps: int = 2) -> str:
        self.memory.add_interaction("user", user_input)
        prompt_input = user_input
        response = ""

        for step in range(max_steps):
            messages = self.build_prompt(prompt_input)
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            response = completion.choices[0].message.content.strip()
            self.memory.add_interaction("assistant", response)

            action_result = self.pattern.parse_response(response)
            if action_result is None:
                return response

            validation_error = self._validate_action(action_result)
            if validation_error is not None:
                if step < max_steps - 1:
                    prompt_input = (
                        f"The previous output was invalid: {validation_error}.\n"
                        "Please respond again using the required format."
                    )
                    continue
                return validation_error

            self.last_selected_skill = action_result.skill
            if action_result.skill:
                self.memory.add_interaction("skill", f"Selected skill: {action_result.skill}")

            tool_result = self.tool_registry.get_tool(action_result.action).function(action_result.input or "")
            self.memory.add_interaction("tool", f"{action_result.action}: {tool_result}")

            if step < max_steps - 1:
                prompt_input = self.pattern.follow_up_prompt(tool_result, action_result)
            else:
                prompt_input = ""

        return response
