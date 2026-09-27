# 📚 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about a PDF document.

The application retrieves relevant information from the PDF using semantic search and then uses a Hugging Face-hosted LLM to generate an answer based only on the retrieved PDF context.

---

## 🚀 Features

- 📄 PDF document loading
- ✂️ Automatic text chunking
- 🧠 Hugging Face sentence embeddings
- 🔎 Semantic similarity search
- 🗃️ Chroma vector database
- 🤖 Hugging Face Inference API
- 💬 Interactive Streamlit chatbot
- 📚 Source page display
- 🛡️ Prevents answers from outside the PDF context
- ❌ Responds when information is not found in the PDF
- 💾 Chat history during the current session

---

## 🏗️ Architecture

```text
                PDF
                 │
                 ▼
          PyPDFLoader
                 │
                 ▼
           PDF Documents
                 │
                 ▼
        Text Chunking
                 │
                 ▼
      Hugging Face Embeddings
                 │
                 ▼
       Chroma Vector Database
                 │
                 ▼
             Retriever
                 │
                 ▼
       Relevant PDF Chunks
                 │
                 ▼
          Context Creation
                 │
                 ▼
       Hugging Face Inference
                 │
                 ▼
          GPT-OSS-120B
                 │
                 ▼
             Answer
                 │
                 ▼
          Source Pages





          🛠️ Technologies Used
Python
Streamlit
LangChain
Hugging Face
Sentence Transformers
Chroma
PyPDF
GPT-OSS-120B