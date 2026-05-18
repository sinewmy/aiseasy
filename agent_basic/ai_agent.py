from typing import Any, Callable, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class SimpleAIAgent:
    def __init__(self, model_name: str = "distilgpt2", device: int = -1) -> None:
        """Create a simple local AI agent using a public Transformer model."""
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self.functions: Dict[str, Dict[str, Any]] = {}

    def register_function(
        self,
        name: str,
        function: Callable[..., Any],
        description: Optional[str] = None,
    ) -> None:
        """Register a custom function that the agent can call later."""
        self.functions[name] = {
            "function": function,
            "description": description or "No description provided.",
        }

    def help_message(self) -> str:
        lines = [
            "Simple AI Agent Help:\n",
            "- Type a question, idea, or prompt and press Enter.",
            "- Use simple natural language to interact with the agent.",
            "- Type 'exit', 'quit', or 'q' to stop.",
            "- Type 'help' or '?' to see this message again.",
        ]
        if self.functions:
            lines.append("\nRegistered functions:")
            for name, info in self.functions.items():
                lines.append(f"- {name}: {info['description']}")
        return "\n".join(lines)

    def generate_reply(
        self,
        prompt: str,
        max_length: int = 120,
        num_return_sequences: int = 1,
    ) -> str:
        """Generate text from the model based on the prompt."""
        output = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return output[0]["generated_text"].strip()

    def handle_user_request(self, user_request: str) -> str:
        """Handle a user request and return an agent response."""
        prompt = user_request.strip()
        if not prompt:
            return "Please type a prompt or question."

        if prompt.lower().startswith("function:"):
            return self._run_function_prompt(prompt)

        return self.generate_reply(prompt)

    def _run_function_prompt(self, prompt: str) -> str:
        """Run a registered function by using the prompt format `function:<name> args...`."""
        parts = prompt.split(maxsplit=1)
        if len(parts) < 1:
            return "Invalid function command. Use: function:<name> [arguments]"

        name_command = parts[0]
        function_name = name_command.split(":", 1)[-1].strip()
        args = parts[1] if len(parts) > 1 else ""

        if function_name not in self.functions:
            return f"Function '{function_name}' is not registered."

        function = self.functions[function_name]["function"]
        try:
            return str(function(args))
        except Exception as error:
            return f"Function '{function_name}' failed: {error}"
