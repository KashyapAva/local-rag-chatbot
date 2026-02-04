from typing import List, Dict, Any, Tuple

def build_context_snippets(hits: List[Tuple[float, Dict[str, Any]]], max_chars: int = 3500) -> str:
    """
    Turn retrieved chunks into a compact context block.
    We use a character budget for now (token budgeting comes later).
    """
    parts = []
    used = 0
    for _, c in hits:
        header = f"[{c['source_file']}#chunk{c['chunk_id']}]"
        body = c["text"].strip()
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


def build_messages(user_question: str, context: str) -> list:
    
    system = (
        "You are a customer support assistant.\n"
        "Use ONLY the provided CONTEXT. Do NOT use external knowledge.\n"
        "Do NOT mention external documentation or APIs unless they appear verbatim in CONTEXT.\n"
        "Do NOT include URLs/links.\n"
        "Do NOT copy CONTEXT headers as standalone lines.\n"
        "If the answer is not in the context, say exactly: \"I don't know based on the provided documents.\"\n"
        "Output format: 3-6 bullets. No paragraphs.\n"
        "Do NOT include citations, source names, file names, or chunk IDs in the answer.\n"
    )


    # Extract allowed citations from context headers (lines like [file#chunkID])
    allowed = []
    for line in context.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]") and "#chunk" in line:
            allowed.append(line)

    allowed_block = "\n".join(allowed[:15])

    user = (
        f"ALLOWED CITATIONS (copy/paste only from this list):\n{allowed_block}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{user_question}\n\n"
        "Answer in 3-6 bullets. Each bullet must end with allowed citations."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

