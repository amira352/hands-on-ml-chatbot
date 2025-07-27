import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(" API Key loaded?", bool(api_key))

#Prevents reloading of models and vector store on every interaction
@st.cache_resource
def load_components():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.load_local(
        "faiss_store",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4o",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=1024,
        temperature=0.2
    )
    
    return embedding_model, vector_store, llm

embedding_model, vector_store, llm = load_components()

st.title("💬 Hands-on-ML chatbot")

# Initialize chat history and memory in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input (chat-style)
if query := st.chat_input("Type your question here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run QA chain with persistent memory
    with st.chat_message("assistant"):
        response = st.session_state.qa_chain({"question": query})["answer"]
        st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()