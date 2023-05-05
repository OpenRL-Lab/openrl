from openrl.runners.common import Chat6BAgent as Agent


def chat():
    agent = Agent.load(
        "THUDM/chatglm-6b",
        device="cuda:0",
    )
    history = []
    print("Welcome to OpenRL!")
    while True:
        input_text = input("> User: ")
        if input_text in ["quit", "exit", "quit()", "exit()", "q"]:
            break
        elif input_text == "reset":
            history = []
            print("Welcome to OpenRL!")
            continue
        response = agent.chat(input_text, history)
        print(f"> Agent: {response}")
        history.append(input_text)
        history.append(response)


if __name__ == "__main__":
    chat()
