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

# k slider (improved default)
k = st.slider(
    "Number of context chunks (k)",
    min_value=2,
    max_value=20,
    value=10,
    step=1,
    help="Higher k = more context but slower"
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
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    documents = splitter.split_documents(docs)

    # 🔹 Better semantic embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = FAISS.from_documents(documents, embeddings)
    return db


def build_chain(db, k):
    retriever = db.as_retriever(search_kwargs={"k": k})

    llm = ChatOllama(
        model="llama3",
        temperature=0,
        num_ctx=8192
    )

    # 🔹 Improved prompt with inference
    prompt = ChatPromptTemplate.from_template("""
You are a technical assistant.

Using ONLY the provided context:
- If the concept is explained indirectly, infer a concise definition
- If formulas are present, include them
- If the concept is truly absent, say "I don't know."
- Do not add external knowledge.

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

    # 🔹 Index only once
    if "db" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Indexing PDF (one-time)..."):
            st.session_state.db = build_vectorstore(pdf_path)

        st.success("PDF indexed successfully!")

    # 🔹 Build chain only if k changes
    if "chain" not in st.session_state or st.session_state.k != k:
        st.session_state.chain = build_chain(st.session_state.db, k)
        st.session_state.k = k

    # 🔹 Handle querying
    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"input": query})

        st.subheader("✅ Answer")
        st.write(response["answer"])

        st.subheader("📚 Sources")
        for i, doc in enumerate(response["context"], 1):
            page = doc.metadata.get("page", "N/A")
            st.write(f"{i}. Page {page}")

