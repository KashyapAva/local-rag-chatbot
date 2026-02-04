import streamlit as st
import requests
import uuid

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("📄 Local RAG Chatbot")

# -------------------------------
# Session state
# -------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Helper: call backend
# -------------------------------
def call_chat_api(message: str):
    payload = {
        "message": message,
        "session_id": st.session_state.session_id,
    }
    resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()

# -------------------------------
# Layout
# -------------------------------
chat_col, sources_col = st.columns([2, 1])

# -------------------------------
# Chat column
# -------------------------------
with chat_col:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # Call backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = call_chat_api(user_input)
                    answer = response["answer"]
                    retrieved = response.get("retrieved", [])
                    trace_id = response.get("trace_id")

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["answer"],
                            "retrieved": response.get("retrieved", []),
                            "sources_used": response.get("sources_used", []),
                        }
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

# -------------------------------
# Sources column
# -------------------------------
with sources_col:
    st.subheader("Sources")

    last_assistant = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        None,
    )

    if last_assistant:
        retrieved = last_assistant.get("retrieved", [])
        if retrieved:
            for i, r in enumerate(retrieved, start=1):
                title = f"[{i}] {r['source_file']}#chunk{r['chunk_id']} (score={r['score']:.2f})"
                with st.expander(title):
                    st.write(r.get("text_preview", ""))
                    if r.get("page_number") is not None:
                        st.caption(f"Page: {r['page_number']}")
        else:
            st.write("No sources to display.")
    else:
        st.write("No sources to display.")