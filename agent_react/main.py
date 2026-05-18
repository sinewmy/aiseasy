from react_agent import create_default_agent


def run_agent() -> None:
    agent = create_default_agent()
    print("ReAct Agent is ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        try:
            answer = agent.run(user_input)
            print("\nAgent:")
            print(answer)
            print()
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    run_agent()
