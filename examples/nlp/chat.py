from openrl.runners.common import ChatAgent as Agent


def chat():
    agent = Agent.load(
        "./ppo_agent",
        tokenizer="/home/huangshiyu/data_server/huggingface/models/gpt2",
    )
    history = []
    print("Welcome to OpenRL!")
    while True:
        input_text = input("> User: ")
        if input_text in ["quit", "exit", "quit()", "exit()", "q"]:
            break
        elif input_text in "reset":
            history = []
            print("Welcome to OpenRL!")
            continue
        response = agent.chat(input_text, history)
        print(f"> OpenRL Agent: {response}")
        history.append(input_text)
        history.append(response)


if __name__ == "__main__":
    chat()
