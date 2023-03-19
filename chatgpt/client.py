#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Literal

import aiohttp
import tiktoken
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape


load_dotenv()


class APIError(Exception):
    pass


class ChatGPTClient:
    MODEL = "gpt-3.5-turbo"

    def __init__(
        self,
        *,
        initial_prompt: str | Literal[False] | None = None,
        max_context_tokens: int = 3072,
        max_response_tokens: int = 500,
        user_id: str | None = None,
    ):
        if not initial_prompt and initial_prompt is not False:
            initial_prompt = "\n".join([
                f"You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.",
                f"Knowledge cutoff: 2021-09",
                f"Current date: {datetime.utcnow().strftime('%Y-%m-%d')}",
            ])

        self._max_context_tokens = max_context_tokens
        self._max_response_tokens = max_response_tokens

        self._user_id = str(user_id or "")

        self._context = []
        if initial_prompt:
            self._context.append({
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
            num_tokens = len(tokenizer.encode("\n\n".join(x["content"] for x in [*self._context, question])))
            if num_tokens > self._max_context_tokens and len(x for x in self._context if x["role"] != "system") > 1:
                if self._context and self._context[0]["role"] == "system":
                    self._context[:] = [self._context[0], *self._context[2:]]
                else:
                    self._context[:] = self._context[1:]
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
                    "messages": [*self._context, question],
                    "max_tokens": self._max_response_tokens,
                    "user": hashlib.sha256(self._user_id.encode()).hexdigest(),
                },
            ) as r:
                res = await r.json()

        if "error" in res:
            raise APIError(res["error"])

        self._context.append(question)

        answer = res["choices"][0]["message"]

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://api.openai.com/v1/moderations",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                },
                json={
                    "input": question["content"],
                },
            ) as r:
                question_res = await r.json()
            if "error" in question_res:
                raise APIError(question_res["error"])

            async with session.post(
                url="https://api.openai.com/v1/moderations",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                },
                json={
                    "input": f"{question['content']}\n\n{answer['content']}",
                },
            ) as r:
                answer_res = await r.json()
            if "error" in answer_res:
                raise APIError(answer_res["error"])

        categories = set()
        for k, v in question_res["results"][0]["categories"].items():
            if v:
                categories.add(k)
        for k, v in answer_res["results"][0]["categories"].items():
            if v:
                categories.add(k)

        self._context.append(answer)

        if answer_res["results"][0]["flagged"]:
            answer["content"] = f"||{answer['content']}||"

        if question_res["results"][0]["flagged"] or answer_res["results"][0]["flagged"]:
            answer["content"] = f":warning: This content has been flagged as **{', '.join(sorted(categories))}**.\n\n{answer['content']}"

        return answer["content"]

    def reset_context(self) -> None:
        if self._context and self._context[0]["role"] == "system":
            self._context[:] = [self._context[0]]
        else:
            self._context.clear()
