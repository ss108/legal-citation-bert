from typing import Dict, List, Literal

import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_OPTS = Literal["gpt-4-turbo-preview", "gpt-4o", "gpt-4o-mini"]


async def chat(
    *,
    system_prompt: str,
    messages: List[Dict],
    model: OPENAI_MODEL_OPTS = "gpt-4-turbo-preview",
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
