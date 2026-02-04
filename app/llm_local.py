import requests
from typing import List, Dict, Any

LOCAL_LLM_URL = "http://127.0.0.1:8001/v1/chat/completions"

def local_chat_completions(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 300,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    r = requests.post(LOCAL_LLM_URL, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()
