"""
rag_pipeline.py
================
Komponen LangChain untuk RAG System:
- Document loading & chunking
- Embedding dengan HuggingFace
- Vector store FAISS
- Retrieval chain
"""

import os
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

load_dotenv()


# ─────────────────────────────────────────────
# 1. LLM: Groq (llama3-8b-8192)
# ─────────────────────────────────────────────
def get_llm():
    """Inisialisasi LLM dari Groq."""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ─────────────────────────────────────────────
# 2. Embedding Model (HuggingFace – lokal, gratis)
# ─────────────────────────────────────────────
def get_embeddings():
    """Inisialisasi embedding model dari HuggingFace."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ─────────────────────────────────────────────
# 3. Document Loader
# ─────────────────────────────────────────────
def load_pdf(file_path: str) -> List[Document]:
    """Load dokumen dari file PDF."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_web(url: str) -> List[Document]:
    """Load dokumen dari URL web."""
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents


# ─────────────────────────────────────────────
# 4. Text Splitter
# ─────────────────────────────────────────────
def split_documents(documents: List[Document]) -> List[Document]:
    """Bagi dokumen menjadi chunk-chunk kecil."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ─────────────────────────────────────────────
# 5. Vector Store (FAISS)
# ─────────────────────────────────────────────
def create_vectorstore(chunks: List[Document]) -> FAISS:
    """Buat vector store dari chunk dokumen."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def load_vectorstore(path: str) -> FAISS:
    """Load vector store yang sudah tersimpan."""
    embeddings = get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def save_vectorstore(vectorstore: FAISS, path: str):
    """Simpan vector store ke disk."""
    vectorstore.save_local(path)


# ─────────────────────────────────────────────
# 6. RAG Prompt Template
# ─────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
Kamu adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.

Gunakan HANYA informasi dari konteks berikut untuk menjawab pertanyaan.
Jika informasi tidak ada di konteks, katakan "Saya tidak menemukan informasi tersebut di dokumen."

Konteks dari dokumen:
{context}

Pertanyaan: {question}

Jawaban (gunakan Bahasa Indonesia yang jelas dan informatif):
""")


# ─────────────────────────────────────────────
# 7. RAG Chain (LangChain LCEL)
# ─────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS):
    """
    Bangun RAG chain menggunakan LangChain Expression Language (LCEL).
    Flow: retrieve → format → prompt → LLM → parse
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"[Halaman {doc.metadata.get('page', '?')+1}]\n{doc.page_content}"
            for doc in docs
        )

    llm = get_llm()
    parser = StrOutputParser()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | parser
    )
    return chain, retriever


# ─────────────────────────────────────────────
# 8. Grader: Apakah dokumen relevan?
# ─────────────────────────────────────────────
GRADE_PROMPT = ChatPromptTemplate.from_template("""
Nilai apakah dokumen berikut RELEVAN dengan pertanyaan yang diberikan.

Dokumen: {document}
Pertanyaan: {question}

Jawab HANYA dengan satu kata: "relevan" atau "tidak_relevan"
""")


def grade_document(document: str, question: str) -> str:
    """Grade apakah dokumen relevan dengan pertanyaan (untuk LangGraph)."""
    llm = get_llm()
    chain = GRADE_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"document": document, "question": question})
    return result.strip().lower()
