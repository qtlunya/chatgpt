#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import os
import uuid
from collections import defaultdict
from datetime import datetime
from getpass import getuser
from typing import Literal

import aiohttp
import tiktoken
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape


load_dotenv()

context = defaultdict(dict)


class APIError(Exception):
    pass


class APIWarning(Warning):
    pass


class ChatGPTClient:
    MODEL = "gpt-3.5-turbo"

    def __init__(self, *, initial_prompt: str | Literal[False] | None = None, user_id: str | None = None):
        if not initial_prompt and initial_prompt is not False:
            initial_prompt = "\n".join([
                f"You are ChatGPT, a large language model trained by OpenAI.",
                f"Knowledge cutoff: 2021-09",
                f"Current date: {datetime.utcnow().strftime('%Y-%m-%d')}",
            ])

        self.user_id = user_id

        self.context = []
        if initial_prompt:
            self.context.append({
                "role": "system",
                "content": initial_prompt,
            })

    async def get_answer(self, prompt: str) -> str:
        question = {
            "role": "user",
            "content": prompt,
        }

        try:
            tokenizer = tiktoken.encoding_for_model(self.MODEL)
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        while True:
            num_tokens = len(tokenizer.encode("\n\n".join(x["content"] for x in [*self.context, question])))
            # Try to leave at least 25% of tokens for the response if possible
            if num_tokens > 3072 and len(x for x in self.context if x["role"] != "system") > 1:
                if self.context[0]["role"] == "system":
                    self.context = [self.context[0], *self.context[2:]]
                else:
                    self.context = self.context[1:]
            else:
                break

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                },
                json={
                    "model": self.MODEL,
                    "messages": [*self.context, question],
                    "user": self.user_id,
                },
            ) as r:
                res = await r.json()

        if "error" in res:
            raise APIError(res["error"])

        self.context.append(question)

        answer = res["choices"][0]["message"]
        self.context.append(answer)
        return answer["content"]
