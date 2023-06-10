#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Literal

import aiohttp
import tiktoken


class APIError(Exception):
    pass


class ChatGPTClient:
    MODEL = "gpt-3.5-turbo"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        initial_prompt: str | Literal[False] | None = None,
        max_context_tokens: int = 3072,
        max_completion_tokens: int = 500,
        user_id: str | None = None,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not initial_prompt and initial_prompt is not False:
            initial_prompt = "\n".join([
                "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.",
                "Knowledge cutoff: 2021-09",
                f"Current date: {datetime.utcnow().strftime('%Y-%m-%d')}",
            ])

        self._max_context_tokens = max_context_tokens
        self._max_completion_tokens = max_completion_tokens

        self._user_id = str(user_id or "")

        self._context = []
        if initial_prompt:
            self._context.append({
                "role": "system",
                "content": initial_prompt,
            })

        self._session = aiohttp.ClientSession()

    def __del__(self):
        self._session.close()

    async def get_completion(self, prompt: str) -> str:
        prompt = {
            "role": "user",
            "content": prompt,
        }

        try:
            tokenizer = tiktoken.encoding_for_model(self.MODEL)
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        while True:
            num_tokens = len(tokenizer.encode("\n\n".join(x["content"] for x in [*self._context, prompt])))
            if num_tokens > self._max_context_tokens and len(x for x in self._context if x["role"] != "system") > 1:
                if self._context and self._context[0]["role"] == "system":
                    self._context[:] = [self._context[0], *self._context[2:]]
                else:
                    self._context[:] = self._context[1:]
            else:
                break

        async with self._session.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "model": self.MODEL,
                "messages": [*self._context, prompt],
                "max_tokens": self._max_completion_tokens,
                "user": hashlib.sha256(self._user_id.encode()).hexdigest(),
            },
        ) as r:
            res = await r.json()

        if "error" in res:
            raise APIError(res["error"])

        self._context.append(prompt)

        completion = res["choices"][0]["message"]

        async with self._session.post(
            url="https://api.openai.com/v1/moderations",
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "input": prompt["content"],
            },
        ) as r:
            prompt_res = await r.json()

        if "error" in prompt_res:
            raise APIError(prompt_res["error"])

        async with self._session.post(
            url="https://api.openai.com/v1/moderations",
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "input": f"{prompt['content']}\n\n{completion['content']}",
            },
        ) as r:
            completion_res = await r.json()

        if "error" in completion_res:
            raise APIError(completion_res["error"])

        categories = set()
        for k, v in prompt_res["results"][0]["categories"].items():
            if v:
                categories.add(k)
        for k, v in completion_res["results"][0]["categories"].items():
            if v:
                categories.add(k)

        self._context.append(completion)

        completion_text = completion["content"]

        if completion_res["results"][0]["flagged"]:
            completion_text = f"||{completion_text}||"

        if prompt_res["results"][0]["flagged"] or completion_res["results"][0]["flagged"]:
            completion_text = (
                f":warning: This content has been flagged as **{', '.join(sorted(categories))}**.\n\n{completion_text}"
            )

        return completion_text

    def reset_context(self) -> None:
        if self._context and self._context[0]["role"] == "system":
            self._context[:] = [self._context[0]]
        else:
            self._context.clear()
