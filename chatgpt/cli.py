import asyncio
import readline  # noqa: F401

import cloup
from cloup.constraints import mutually_exclusive

from . import ChatGPTClient


@cloup.command()
@cloup.option("-i", "--initial-prompt", type=str, help="The initial prompt to use for ChatGPT.")
@cloup.option("-n", "--no-initial-prompt",  is_flag=True, help="Suppress the default initial prompt for ChatGPT.")
@cloup.constraint(mutually_exclusive, ["initial_prompt", "no_initial_prompt"])
def cli(initial_prompt: str | None, no_initial_prompt: bool) -> None:
    client = ChatGPTClient(initial_prompt=False if no_initial_prompt else initial_prompt)
    while True:
        try:
            prompt = input("> ")
            print(asyncio.run(client.get_answer(prompt)))
        except (EOFError, KeyboardInterrupt):
            break
