from typing import Optional

from base_harness import BaseAgentHarness
from pattern import ReActPattern


class ReActAgentHarness(BaseAgentHarness):
    """A ReAct-style harness that uses the ReAct pattern for model interactions."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(pattern=ReActPattern(), api_key=api_key)


def create_default_agent() -> ReActAgentHarness:
    """Create a default agent with tools and skills loaded from directories."""
    return ReActAgentHarness()
