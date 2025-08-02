# Hands-On Machine Learning RAG Chatbot 🤖📚

A local chatbot powered by Retrieval-Augmented Generation (RAG) that answers questions from the **"Hands-On Machine Learning"** book using FAISS and LangChain.

---

## Features

- Question answering based on book content
- Uses FAISS for efficient similarity search
- LangChain for chaining LLM + retriever
- Streamlit frontend for interaction
- HuggingFace `all-MiniLM-L6-v2` for embeddings
- OpenAI (or OpenRouter) as the LLM backend

---

## 📁Project Structure

```
├── app.py # Streamlit app
├── faiss_store/ # FAISS index & metadata
│ ├── index.faiss
│ └── index.pkl
├── requirements.txt
└── README.md
```
