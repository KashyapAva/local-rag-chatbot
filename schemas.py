from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    conversation_id: str = "default"
    message: str
    # generation params (LLM hyperparameters)
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 300

class Citation(BaseModel):
    source_file: str
    page_number: Optional[int] = None
    chunk_id: int
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    answer: str
    sources_used: List[str]
    retrieved: List[Citation]
