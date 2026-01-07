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
st.write("Ask questions about your PDF")

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
def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documents = splitter.split_documents(docs)

    db = FAISS.from_documents(
        documents,
        OllamaEmbeddings()
    )
    return db


def build_chain(db, k):
    retriever = db.as_retriever(search_kwargs={"k": k})

    llm = ChatOllama(
        model="llama3:8b",
        temperature=0,
        num_ctx=8192
    )

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
    return create_retrieval_chain(retriever, document_chain)


# ---------------- LOGIC ----------------
if uploaded_file:

    # Index only once per PDF
    if "db" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Indexing PDF (one-time)..."):
            st.session_state.db = build_vectorstore(pdf_path)

        st.success("PDF indexed successfully!")

    # Ask questions without re-indexing
    if query:
        chain = build_chain(st.session_state.db, k)

        with st.spinner("Thinking..."):
            response = chain.invoke({"input": query})

        st.subheader("✅ Answer")
        st.write(response["answer"])

        st.subheader("📚 Sources")
        for i, doc in enumerate(response["context"], 1):
            page = doc.metadata.get("page", "N/A")
            st.write(f"{i}. Page {page}")

