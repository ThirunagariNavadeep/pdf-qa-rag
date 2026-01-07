import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


# ---------------- UI ----------------
st.set_page_config(page_title="PDF Q&A with Ollama", layout="wide")
st.title("📄 PDF Question Answering (RAG)")
st.write("Ask questions about your PDF using **Ollama + LangChain**")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# 🔹 NEW: Slider for k value
k = st.slider(
    "Number of context chunks (k)",
    min_value=2,
    max_value=20,
    value=4,
    step=1,
    help="Higher k = more pages considered, but slower"
)

query = st.text_input(
    "Ask a question",
    placeholder="What is Scaled Dot-Product Attention?"
)


# ---------------- RAG PIPELINE ----------------
@st.cache_resource(show_spinner=True)
def build_rag(pdf_path, k):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documents = splitter.split_documents(docs)

    # Vector store
    db = FAISS.from_documents(
        documents,
        OllamaEmbeddings()
    )

    # 🔹 USE k HERE
    retriever = db.as_retriever(search_kwargs={"k": k})

    # LLM
    llm = ChatOllama(
        model="llama2",
        temperature=0,
        num_ctx=4096
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a technical assistant.
Answer ONLY using the provided context.
If the answer is not present, say "I don't know."

<context>
{context}
</context>

Question: {input}

Answer:
""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return retrieval_chain


# ---------------- LOGIC ----------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Indexing PDF..."):
        rag_chain = build_rag(pdf_path, k)  # 🔹 pass k

    st.success(f"PDF indexed successfully! (k = {k})")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": query})

        st.subheader("✅ Answer")
        st.write(response["answer"])

        st.subheader("📚 Sources")
        for i, doc in enumerate(response["context"], 1):
            page = doc.metadata.get("page", "N/A")
            st.write(f"{i}. Page {page}")
