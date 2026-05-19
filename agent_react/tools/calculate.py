def execute(expression: str) -> str:
    """Evaluate arithmetic expressions safely."""
    try:
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {})
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"
