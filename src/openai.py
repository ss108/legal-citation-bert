from typing import Dict, List

import openai
import tiktoken
from dotenv import load_dotenv

load_dotenv()


def get_logit_bias():
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode("In")
    print(tokens)


async def chat(
    *,
    system_prompt: str,
    messages: List[Dict],
    model: str = "gpt-4-turbo",
    temperature: float = 1.0,
) -> str:
    client = openai.AsyncOpenAI()
    system_message = {"role": "system", "content": system_prompt}

    resp = await client.chat.completions.create(
        model=model,
        messages=[system_message] + messages,  # pyright: ignore
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    answer = resp.choices[0].message.content if len(resp.choices) > 0 else ""

    return answer  # pyright: ignore
