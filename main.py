import os
from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse, Citation
from app.retriever import Retriever
from app.prompting import build_context_snippets, build_messages
from app.llm_local import local_chat_completions
from app.postprocess import clean_answer
from fastapi import HTTPException
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(PROJECT_ROOT, "rag_index")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Change this to whatever model you pulled in Ollama
LOCAL_MODEL_NAME = "llama3.2"

app = FastAPI(title="LLM RAG Chatbot (RAG + Local LLM)")

retriever: Retriever | None = None

@app.on_event("startup")
def load_index():
    global retriever
    retriever = Retriever(index_dir=INDEX_DIR, embed_model=EMBED_MODEL_NAME)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
  try:
    assert retriever is not None, "Retriever not loaded"
    
    RETRIEVAL_TOP_K = 3

    hits = retriever.search(req.message, top_k=RETRIEVAL_TOP_K)
    
    min_score = 0.50
    if (not hits) or (hits[0][0] < min_score):
        return ChatResponse(
            conversation_id=req.conversation_id,
            message=req.message,
            answer="I don't know based on the provided documents.",
            sources_used=[],
            retrieved=[]
        )


    # Build retrieval payload (for response + citations)
    retrieved = []
    sources_used = []


    for score, c in hits:
        preview = (c["text"][:240] + "…").replace("\n", " ")
        retrieved.append(Citation(
            source_file=c["source_file"],
            page_number=c.get("page_number"),
            chunk_id=c["chunk_id"],
            score=score,
            text_preview=preview
        ))
        
    for _, c in hits:
        tag = f"{c['source_file']}#chunk{c['chunk_id']}"
        if tag not in sources_used:
            sources_used.append(tag)

    # Prompt builder
    context = build_context_snippets(hits, max_chars=2000)
    messages = build_messages(req.message, context)

    # Local LLM
    
    answer = local_chat_completions(
    model="phi3-mini",
    messages=messages,
    temperature=req.temperature,
    top_p=req.top_p,
    max_tokens=req.max_tokens,
    )
    
    return ChatResponse(
        conversation_id=req.conversation_id,
        message=req.message,
        answer = clean_answer(answer),
        sources_used=sources_used[:5],
        retrieved=retrieved
    )
    
  except Exception as e:
        tb = traceback.format_exc()
        print(tb)  # shows in uvicorn terminal
        raise HTTPException(status_code=500, detail=str(e))

