# pdf-qa-rag

# 📄 PDF Q&A Using RAG

A **PDF Question Answering system** built using **Retrieval-Augmented Generation (RAG)**.  
This application allows users to upload a PDF, ask questions about its content, and receive **accurate, context-grounded answers** along with **source page references**.

The project uses **Ollama (local LLMs)**, **LangChain**, **FAISS**, and **Streamlit**.

---

## 🚀 Features

- 📄 Upload any PDF document
- 🔍 Semantic search across the entire PDF
- 🤖 Accurate answers using Retrieval-Augmented Generation (RAG)
- 📚 Page-level source citations
- 🎛️ Dynamic control over context size (`k` value)
- ⚡ Fast responses after one-time indexing
- 🖥️ Simple and clean Streamlit UI
- 🔒 Fully local inference using Ollama (no API keys required)

---

## 🧠 How It Works (RAG Pipeline)

1. **PDF Loading** – PDF is read page by page
2. **Text Chunking** – Pages are split into overlapping chunks
3. **Embeddings** – Chunks are converted into vector embeddings
4. **Vector Store** – FAISS indexes all embeddings
5. **Retrieval** – Top `k` relevant chunks are selected
6. **Generation** – LLM answers using only retrieved context

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI
- **LangChain** – RAG framework
- **FAISS** – Vector similarity search
- **Ollama** – Local LLM inference
- **PyPDF** – PDF parsing

---

## 📂 Project Structure

```text
pdf-qa-rag/
│
├── rc_app.py              # Streamlit application
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── sample.pdf             # (Optional) Sample PDF
