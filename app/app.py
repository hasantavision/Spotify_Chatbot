import os
from typing import List, Tuple

import streamlit as st
from rag_functionality import eval_func, rag_func

# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    has_key = openai_api_key.startswith("sk-")
    eval_metrics = st.toggle(
        label="Use Eval metrics", value=True, disabled=not has_key
    )
    use_openai = st.toggle(label="Use OpenAI", value=False)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello there, ask me anything regarding Spotify's reviews.",
        }
    ]
# Tracks (question, answer) pairs for conversation-aware retrieval
if "chat_history" not in st.session_state:
    st.session_state["chat_history"]: List[Tuple[str, str]] = []

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------------------------------------------------------
# Handle new user input
# ---------------------------------------------------------------------------
user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        if use_openai and not has_key:
            st.warning("Please enter your OpenAI API key!", icon="⚠")
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    ai_response: dict = {}
    is_error = False
    word_limit = False

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if len(user_prompt) > 500:
                st.write("Your question is too long — keep it under 500 characters.")
                word_limit = True
            else:
                try:
                    ai_response = rag_func(
                        user_prompt,
                        use_openai,
                        chat_history=st.session_state["chat_history"],
                    )
                    answer = ai_response["answer"]
                    source_count = len(ai_response.get("context", []))
                    st.write(answer)
                    st.caption(f"Sources retrieved: {source_count}")
                except Exception as exc:
                    is_error = True
                    answer = str(exc)
                    st.error(answer)

    # Persist the Q&A pair in chat history for subsequent turns
    if not word_limit and not is_error:
        st.session_state["chat_history"].append((user_prompt, answer))
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Optional evaluation block
    if eval_metrics and not word_limit and not is_error and has_key:
        with st.chat_message("assistant"):
            with st.spinner("Evaluating…"):
                try:
                    retrieval_context = [
                        doc.page_content for doc in ai_response.get("context", [])
                    ]
                    metric_results = eval_func(user_prompt, answer, retrieval_context)
                    eval_lines = []
                    for name, score, reason in metric_results:
                        eval_lines.append(f"**{name}**: {score:.2f}  \n{reason}")
                    eval_text = "\n\n".join(eval_lines)
                    st.write(eval_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": eval_text}
                    )
                except Exception as exc:
                    st.error(str(exc))
