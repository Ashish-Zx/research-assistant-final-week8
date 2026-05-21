# agent/tools/registry.py
import inspect
from typing import Callable, get_type_hints
from pydantic import create_model


class Tool:
    """A single tool with automatic JSON Schema generation."""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
        # Generate schema from function signature
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        fields = {}
        for param_name, param in sig.parameters.items():
            annotation = hints.get(param_name, str)
            default = (
                param.default if param.default is not inspect.Parameter.empty else ...
            )
            fields[param_name] = (annotation, default)
        self.model = create_model(f"{name}_args", **fields)
        self.schema = self.model.model_json_schema()

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_tools_list_for_api(self) -> list[dict]:
        return [tool.to_openai_tool() for tool in self._tools.values()]
