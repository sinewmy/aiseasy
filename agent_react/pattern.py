from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ActionResult:
    skill: Optional[str]
    action: Optional[str]
    input: Optional[str]
    raw: str


class AgentPattern(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def system_instructions(self) -> str:
        return ""

    @property
    def system_prompt_file(self) -> Optional[Path]:
        return None

    @abstractmethod
    def format_prompt(
        self,
        base_messages: List[Dict[str, str]],
        tool_text: str,
        skill_text: str,
    ) -> List[Dict[str, str]]:
        ...

    @abstractmethod
    def parse_response(self, response: str) -> Optional[ActionResult]:
        ...

    @abstractmethod
    def follow_up_prompt(self, tool_result: str, action: ActionResult) -> str:
        ...


class ReActPattern(AgentPattern):
    name = "react"

    @property
    def system_prompt_file(self) -> Optional[Path]:
        return Path(__file__).resolve().parent / "react_system_prompt.md"

    def format_prompt(
        self,
        base_messages: List[Dict[str, str]],
        tool_text: str,
        skill_text: str,
    ) -> List[Dict[str, str]]:
        messages = list(base_messages)
        if tool_text:
            messages.insert(-1, {"role": "system", "content": tool_text})
        if skill_text:
            messages.insert(-1, {"role": "system", "content": skill_text})
        return messages

    def parse_response(self, response: str) -> Optional[ActionResult]:
        import re

        action_regex = re.compile(
            r"Selected skill:\s*(.+?)\nAction:\s*(.+?)\nAction Input:\s*(.+)",
            re.S,
        )
        match = action_regex.search(response)
        if not match:
            return None
        return ActionResult(
            skill=match.group(1).strip(),
            action=match.group(2).strip(),
            input=match.group(3).strip(),
            raw=response,
        )

    def follow_up_prompt(self, tool_result: str, action: ActionResult) -> str:
        return (
            f"Tool output for {action.action}: {tool_result}\n"
            "Please continue your reasoning and provide the final answer."
        )
