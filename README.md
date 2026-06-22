# 🧠 RAG System — UAS Natural Language Processing

> **Sistem tanya-jawab berbasis dokumen** menggunakan LangChain, LangGraph, dan LangSmith.  
> Proyek Ujian Akhir Semester Mata Kuliah NLP.

---

## 📋 Deskripsi Proyek

RAG System (Retrieval-Augmented Generation) adalah sistem AI yang memungkinkan pengguna untuk **mengunggah dokumen PDF atau memasukkan URL web**, lalu **mengajukan pertanyaan** seputar isi dokumen tersebut. Sistem akan secara otomatis mencari bagian dokumen yang relevan dan menghasilkan jawaban menggunakan LLM.

### Alur Kerja Sistem

```
User Upload PDF/URL
        │
        ▼
[LangChain] Load → Split → Embed → FAISS Vector Store
        │
        ▼
User Ajukan Pertanyaan
        │
        ▼
[LangGraph] StateGraph
   ┌─────┴──────────────────────────────────┐
   │                                        │
   ▼                                        │
Node: retrieve                              │
   │ (cari top-4 dokumen mirip)             │
   ▼                                        │
Node: grade_documents                       │
   │ (filter dokumen relevan)               │
   ├── ada relevan ──► Node: generate       │
   └── tidak ada ───► Node: fallback        │
              │              │              │
              └──────┬───────┘              │
                     ▼                      │
             [LangSmith] Auto-trace ◄───────┘
                     │
                     ▼
             Jawaban ke User
```

---

## 🛠️ Teknologi yang Digunakan

| Library | Versi | Fungsi |
|---------|-------|--------|
| **LangChain** | 0.3.x | Document loading, text splitting, embedding, RAG chain (LCEL) |
| **LangGraph** | 0.4.x | Agentic workflow dengan StateGraph dan conditional edges |
| **LangSmith** | 0.3.x | Tracing, monitoring, dan evaluasi semua langkah pipeline |
| Groq (llama3-8b) | — | LLM untuk generate jawaban (gratis, cepat) |
| HuggingFace Embeddings | — | all-MiniLM-L6-v2 untuk embedding dokumen |
| FAISS | — | Vector store untuk similarity search |
| Streamlit | 1.x | Web UI interaktif |

---

## ✨ Fitur Utama

- 📄 **Upload PDF** atau masukkan **URL web** sebagai sumber dokumen
- 🔍 **Semantic Search** — menemukan chunk paling relevan dengan pertanyaan
- 📊 **Document Grading** — filter otomatis dokumen yang tidak relevan
- 🕸️ **LangGraph Workflow** — pipeline agentic dengan 4 node (retrieve → grade → generate → fallback)
- 🔭 **LangSmith Tracing** — semua run otomatis ter-log dan bisa dimonitor
- 💬 **Multi-turn Chat** — riwayat percakapan tersimpan dalam sesi
- 🌐 **Web Interface** — UI modern dengan Streamlit

---

## 📸 Screenshot

> *(Tambahkan screenshot aplikasi di sini setelah menjalankan)*

| Halaman Utama | Chat dengan Dokumen | LangSmith Dashboard |
|:---:|:---:|:---:|
| ![home](screenshots/home.png) | ![chat](screenshots/chat.png) | ![langsmith](screenshots/langsmith.png) |

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/rag-system-uas-nlp.git
cd rag-system-uas-nlp
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Konfigurasi API Keys

Salin file `.env.example` menjadi `.env`:

```bash
cp .env.example .env
```

Isi file `.env` dengan API key kamu:

```env
GROQ_API_KEY=gsk_...          # https://console.groq.com (gratis)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...    # https://smith.langchain.com (gratis)
LANGCHAIN_PROJECT=rag-system-uas-nlp
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

> 💡 API key juga bisa diisi langsung di sidebar aplikasi tanpa perlu file `.env`.

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📁 Struktur Proyek

```
rag-system-uas-nlp/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── .env.example            # Template environment variables
├── README.md
├── src/
│   ├── __init__.py
│   ├── rag_pipeline.py     # LangChain: loader, splitter, embeddings, chain
│   └── graph_workflow.py   # LangGraph: StateGraph, nodes, edges
└── screenshots/
    ├── home.png
    ├── chat.png
    └── langsmith.png
```

---

## 🔍 Penjelasan Komponen

### LangChain (`src/rag_pipeline.py`)
- **PyPDFLoader / WebBaseLoader** — memuat dokumen dari PDF atau URL
- **RecursiveCharacterTextSplitter** — membagi dokumen menjadi chunk 1000 karakter dengan overlap 200
- **HuggingFaceEmbeddings** — mengkonversi teks ke vektor menggunakan model `all-MiniLM-L6-v2`
- **FAISS** — menyimpan dan mencari vektor secara efisien
- **RAG Chain (LCEL)** — `retriever | format_docs | prompt | llm | parser`
- **ChatGroq** — LLM llama3-8b-8192 dari Groq

### LangGraph (`src/graph_workflow.py`)
- **RAGState** — TypedDict sebagai state yang dibagikan antar node
- **node_retrieve** — mengambil top-4 dokumen dari vector store
- **node_grade_documents** — menilai relevansi setiap dokumen dengan pertanyaan
- **node_generate** — menghasilkan jawaban menggunakan dokumen relevan
- **node_fallback** — memberikan respons default jika tidak ada dokumen relevan
- **decide_after_grading** — conditional edge untuk routing

### LangSmith
- Aktif otomatis ketika `LANGCHAIN_TRACING_V2=true` di `.env`
- Semua chain dan graph run akan ter-trace ke dashboard LangSmith
- Bisa dimonitor di `https://smith.langchain.com`

---

## 📦 Requirements

- Python 3.10+
- Groq API Key (gratis di [console.groq.com](https://console.groq.com))
- LangSmith API Key (gratis di [smith.langchain.com](https://smith.langchain.com))
- Koneksi internet (untuk download model embedding pertama kali)

---
## 📷 Screenshot Aplikasi
### Halaman Utama
![Home](screenshots/home.png)
### Chat dengan Dokumen
![Chat](screenshots/chat.png)
### Proses LangGraph
![LangGraph](screenshots/langsmith.png)

## 👨‍💻 Author

**[MULIADI]**  
NPM: [233510205]  
Program Studi: [Teknik Informatika]  
Mata Kuliah: Natural Language Processing  


---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik — Ujian Akhir Semester NLP.
