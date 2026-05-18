from ai_agent import SimpleAIAgent


def main() -> None:
    agent = SimpleAIAgent()
    print("AI Easy Agent")
    print("Type a question or command, then press Enter.")
    print("Type 'help' for a quick guide, or 'exit' to quit.")

    while True:
        request = input("\nYour request> ").strip()
        if not request:
            continue
        if request.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if request.lower() in {"help", "?"}:
            print(agent.help_message())
            continue

        response = agent.handle_user_request(request)
        print("\n" + response)


if __name__ == "__main__":
    main()
