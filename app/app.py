import os

import streamlit as st
from rag_functionality import rag_func, eval_func


with st.sidebar:
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    os.environ["OPENAI_API_KEY"] = openai_api_key
    eval_metrics = st.toggle(label="Use Eval metrics", value=True, disabled=False if openai_api_key != '' else True)
    use_openai = st.toggle(label="Use OpenAI", value=False)

if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello there, ask me anything regarding Spotify's reviews."}
    ]
if "messages" in st.session_state.keys():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )
    with st.chat_message("user"):
        if use_openai and not openai_api_key.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        is_error = False
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    ai_response = {}
    score = 0
    reason = ""

    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            if len(user_prompt) > 500:
                st.write("Your question is too long, make sure it is less than 500 characters")
                word_limit = True
            else:
                word_limit = False
                try:
                    ai_response = rag_func(user_prompt, use_openai)
                except Exception as e:
                    is_error = True
                    ai_response['result'] = e
                st.write(ai_response['result'])
    if eval_metrics and not word_limit and not is_error and openai_api_key.startswith('sk-'):
        with st.chat_message("assistant"):
            with st.spinner("Evaluating"):
                try:
                    retrieval_context = [doc.page_content for doc in ai_response['source_documents']]
                    score, reason = eval_func(user_prompt, ai_response['result'], retrieval_context)
                    st.write(f"Relevancy Score: {score}\n\n{reason}")
                except Exception as e:
                    st.write(e)
    new_ai_message = {"role": "assistant", "content": ai_response['result']}
    eval_message = {"role": "assistant", "content": f"Relevancy Score: {score}\n\n{reason}"}
    st.session_state.messages.append(new_ai_message)
    st.session_state.messages.append(eval_message)
