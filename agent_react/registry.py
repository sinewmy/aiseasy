import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml


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


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, AgentTool] = {}

    def register_tool(self, tool: AgentTool) -> None:
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[AgentTool]:
        return self.tools.get(name)

    def get_descriptions(self) -> str:
        if not self.tools:
            return ""
        return "Tool descriptions:\n" + "\n".join(
            f"{tool.name}: {tool.description}" for tool in self.tools.values()
        )

    def load_from_directory(self, tools_dir: Optional[Path] = None) -> None:
        if tools_dir is None:
            tools_dir = Path(__file__).resolve().parent / "tools"
        if not tools_dir.exists():
            return

        for yaml_file in sorted(tools_dir.glob("*.yaml")):
            tool_name = yaml_file.stem
            py_file = tools_dir / f"{tool_name}.py"
            if not py_file.exists():
                continue

            try:
                metadata = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if not metadata or "name" not in metadata or "description" not in metadata:
                    continue
            except Exception:
                continue

            try:
                spec = importlib.util.spec_from_file_location(tool_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if not hasattr(module, "execute"):
                    continue
                tool = AgentTool(
                    name=metadata["name"],
                    description=metadata["description"],
                    function=module.execute,
                )
                self.register_tool(tool)
            except Exception:
                continue


class SkillRegistry:
    def __init__(self) -> None:
        self.skills: Dict[str, AgentSkill] = {}

    def register_skill(self, skill: AgentSkill) -> None:
        self.skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[AgentSkill]:
        return self.skills.get(name)

    def get_descriptions(self) -> str:
        if not self.skills:
            return ""
        return "AI Skills:\n" + "\n".join(
            f"{skill.name}: {skill.description} (Preferred tool: {skill.tool_name or 'none'})"
            for skill in self.skills.values()
        )

    def load_from_directory(self, skills_dir: Optional[Path] = None) -> None:
        if skills_dir is None:
            skills_dir = Path(__file__).resolve().parent / "skills"
        if not skills_dir.exists():
            return

        for md_file in sorted(skills_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            if not content.startswith("---"):
                continue
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            try:
                frontmatter = yaml.safe_load(parts[1])
                description = parts[2].strip()
            except Exception:
                continue
            if not frontmatter or "name" not in frontmatter:
                continue
            skill = AgentSkill(
                name=frontmatter["name"],
                description=description,
                keywords=frontmatter.get("keywords", []),
                tool_name=frontmatter.get("tool_name"),
            )
            self.register_skill(skill)
