import json
import os
import time
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()


async def chat(
    *,
    system_prompt: str,
    messages: List[ChatMessage],
    model: str = "mistral-large-latest",
    temperature: float = 0.3,
) -> str:
    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralAsyncClient(api_key=api_key)

    system_message = ChatMessage(content=system_prompt, role="system")
    all_messages = [system_message] + messages

    resp = await client.chat(
        messages=all_messages,
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    answer = resp.choices[0].message.content if len(resp.choices) > 0 else ""
    return answer  # pyright: ignore
