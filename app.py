import streamlit as st
from rag_pipeline import get_rag_chain 

st.title("A Finance/Law QA Chatbot")
st.subheader("The bot answers your questions related to law and finance if it is part of the database")

@st.cache_resource
def load_chain():
    return get_rag_chain()

rag_chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input("Ask a question about your documents..."): 
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(prompt)
    st.session_state.messages.append({'role': 'assistant', 'content': answer})
    with st.chat_message('assistant'):
        st.markdown(answer)


