import os
from typing import List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI  # New SDK requires this import style

# Automatically load environment variables from `.env` file
load_dotenv()

# Read OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in environment variables or .env file.")

# New SDK client
client = OpenAI(api_key=api_key)

DEFAULT_MODEL = "gpt-4.1-nano"

def call_gpt(
    prompt: str,
    system_prompt: str = "You are a helpful assistant for causal reasoning in autonomous driving.",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stop: Optional[Union[str, List[str]]] = None,
) -> str:
    """
    Call OpenAI's ChatCompletion API with given prompt.
    Returns a string response (non-streaming).
    """
    try:
        # New API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[OpenAI API Error] {e}")
        return ""
