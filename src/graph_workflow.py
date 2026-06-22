"""
graph_workflow.py
==================
Komponen LangGraph untuk RAG System:
- State definition
- Node functions
- Conditional edges
- Graph compilation

Flow:
  retrieve → grade_documents → generate
                  ↓ (jika tidak ada yang relevan)
             fallback_response
"""

from typing import TypedDict, List, Annotated
import operator

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

from src.rag_pipeline import build_rag_chain, grade_document, get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────────
# 1. State Definition (LangGraph)
# ─────────────────────────────────────────────
class RAGState(TypedDict):
    """
    State yang digunakan di seluruh graph.
    Setiap node bisa membaca dan menulis ke state ini.
    """
    question: str                          # Pertanyaan dari user
    documents: List[Document]              # Dokumen yang di-retrieve
    relevant_documents: List[Document]     # Dokumen yang lulus grading
    generation: str                        # Jawaban final
    steps: Annotated[List[str], operator.add]  # Log langkah-langkah
    has_relevant_docs: bool                # Flag: ada dokumen relevan?


# ─────────────────────────────────────────────
# 2. Node Functions
# ─────────────────────────────────────────────
def node_retrieve(state: RAGState, vectorstore: FAISS) -> dict:
    """
    Node 1: Retrieve dokumen dari vector store.
    Mengambil top-4 dokumen paling mirip dengan pertanyaan.
    """
    question = state["question"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    documents = retriever.invoke(question)

    return {
        "documents": documents,
        "steps": [f"🔍 Retrieve: ditemukan {len(documents)} dokumen"],
    }


def node_grade_documents(state: RAGState) -> dict:
    """
    Node 2: Grade setiap dokumen — apakah relevan?
    Hanya dokumen relevan yang diteruskan ke node generate.
    """
    question = state["question"]
    documents = state["documents"]
    relevant_docs = []
    grades = []

    for doc in documents:
        grade = grade_document(doc.page_content[:500], question)
        if "relevan" in grade and "tidak" not in grade:
            relevant_docs.append(doc)
            grades.append("✅")
        else:
            grades.append("❌")

    has_relevant = len(relevant_docs) > 0

    return {
        "relevant_documents": relevant_docs,
        "has_relevant_docs": has_relevant,
        "steps": [
            f"📊 Grading: {grades.count('✅')}/{len(documents)} dokumen relevan"
        ],
    }


def node_generate(state: RAGState, vectorstore: FAISS) -> dict:
    """
    Node 3: Generate jawaban menggunakan RAG chain.
    Menggunakan dokumen relevan sebagai konteks.
    """
    question = state["question"]
    relevant_docs = state["relevant_documents"]

    # Format context dari dokumen relevan
    context = "\n\n".join(
        f"[Halaman {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
        for doc in relevant_docs
    )

    # Build prompt & chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template("""
Kamu adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen.

Konteks dari dokumen:
{context}

Pertanyaan: {question}

Jawaban (gunakan Bahasa Indonesia yang jelas):
""")
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    return {
        "generation": generation,
        "steps": ["💬 Generate: jawaban berhasil dibuat"],
    }


def node_fallback(state: RAGState) -> dict:
    """
    Node 4 (Fallback): Tidak ada dokumen relevan ditemukan.
    Berikan jawaban default.
    """
    return {
        "generation": (
            "Maaf, saya tidak menemukan informasi yang relevan dengan pertanyaan Anda "
            "di dalam dokumen yang diunggah. Coba reformulasikan pertanyaan atau "
            "pastikan dokumen mengandung informasi tersebut."
        ),
        "steps": ["⚠️ Fallback: tidak ada dokumen relevan, menggunakan fallback response"],
    }


# ─────────────────────────────────────────────
# 3. Conditional Edge
# ─────────────────────────────────────────────
def decide_after_grading(state: RAGState) -> str:
    """
    Router: setelah grading, ke mana graph harus lanjut?
    - Ada dokumen relevan → node_generate
    - Tidak ada → node_fallback
    """
    if state.get("has_relevant_docs", False):
        return "generate"
    return "fallback"


# ─────────────────────────────────────────────
# 4. Build Graph
# ─────────────────────────────────────────────
def build_rag_graph(vectorstore: FAISS):
    """
    Compile LangGraph workflow.
    Menggabungkan semua node dan edge menjadi satu graph.
    """
    # Bungkus node agar bisa menerima vectorstore
    def retrieve(state):
        return node_retrieve(state, vectorstore)

    def generate(state):
        return node_generate(state, vectorstore)

    # Inisialisasi StateGraph
    workflow = StateGraph(RAGState)

    # Tambahkan nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", node_grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("fallback", node_fallback)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Tambahkan edges
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    # Compile graph
    app = workflow.compile()
    return app


# ─────────────────────────────────────────────
# 5. Run Graph
# ─────────────────────────────────────────────
def run_rag_graph(app, question: str) -> dict:
    """
    Jalankan graph dengan pertanyaan dari user.
    Return: hasil akhir state (answer + steps).
    """
    initial_state = {
        "question": question,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "steps": [],
        "has_relevant_docs": False,
    }
    result = app.invoke(initial_state)
    return result
