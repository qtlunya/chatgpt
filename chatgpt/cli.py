import asyncio
import readline  # noqa: F401
from functools import wraps

import cloup
from cloup.constraints import mutually_exclusive
from dotenv import load_dotenv

from . import ChatGPTClient


def coroutine(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@cloup.command()
@cloup.option("-i", "--initial-prompt", type=str, help="The initial prompt to use for ChatGPT.")
@cloup.option("-n", "--no-initial-prompt",  is_flag=True, help="Suppress the default initial prompt for ChatGPT.")
@cloup.constraint(mutually_exclusive, ["initial_prompt", "no_initial_prompt"])
@coroutine
async def cli(initial_prompt: str | None, no_initial_prompt: bool) -> None:
    load_dotenv()

    client = ChatGPTClient(initial_prompt=False if no_initial_prompt else initial_prompt)

    while True:
        try:
            prompt = input("> ")
            print(await client.get_completion(prompt))
        except (EOFError, KeyboardInterrupt):
            break
