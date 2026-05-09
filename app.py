import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="📚 College RAG Chatbot")

st.title("📚 Multi-Document College Assistant")
st.write("Upload PDFs and ask questions 👇")

# -----------------------------
# Sidebar Upload
# -----------------------------
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# -----------------------------
# Load LLM
# -----------------------------
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

# -----------------------------
# Process PDFs
# -----------------------------
@st.cache_resource
def process_pdfs(files):
    documents = []

    for file in files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file.name

        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    splits = text_splitter.split_documents(documents)

    embeddings = FakeEmbeddings(size=384)

    db = FAISS.from_documents(splits, embeddings)

    return db

# -----------------------------
# Build DB
# -----------------------------
if uploaded_files:
    db = process_pdfs(uploaded_files)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    st.success("✅ PDFs processed successfully!")

    # -----------------------------
    # Chat Input
    # -----------------------------
    query = st.chat_input("Ask your question...")

    if query:
        st.chat_message("user").write(query)

        docs = retriever.invoke(query)

        context = ""
        source_dict = {}

        for doc in docs:
            context += doc.page_content + "\n"

            file = doc.metadata["source"]
            page = doc.metadata.get("page", "N/A")

            if file not in source_dict:
                source_dict[file] = []

            source_dict[file].append(page)

        prompt = f"""
        You are a helpful college assistant.

        Answer ONLY using the context below.

        Context:
        {context}

        Question:
        {query}

        Mention sources clearly.
        """

        try:
            response = llm.invoke(prompt)

            st.chat_message("assistant").write(response.content)

            st.write("📌 Sources:")

            for file, pages in source_dict.items():
                unique_pages = sorted(set(pages))
                pages_str = ", ".join(map(str, unique_pages))

                st.write(f"- {file} (Pages: {pages_str})")

        except Exception:
            st.error("⚠️ Error generating response")

else:
    st.info("📂 Please upload at least one PDF to start")