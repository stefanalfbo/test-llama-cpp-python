import os

from llama_cpp import Llama

llama = Llama(model_path="llama-2-7b-chat.ggmlv3.q4_1.bin", verbose=False)


def get_reply(prompt):
    """Local inference with llama-cpp-python"""
    response = llama(
        f"""Q: {prompt} A:""", max_tokens=64, stop=["Q:", "\n"], echo=False
    )

    return response["choices"].pop()["text"].strip()


def clear():
    """Clears the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def main():
    """The prompt loop."""
    clear()

    while True:
        cli_prompt = input("You: ")

        if cli_prompt == "exit":
            break
        else:
            answer = get_reply(cli_prompt)

            print(f"""Llama: {answer}""")


if __name__ == "__main__":
    main()
