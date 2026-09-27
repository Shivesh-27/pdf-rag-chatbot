import os

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token not found. Please add HF_TOKEN to your .env file.")
    st.stop()


# ============================================================
# 2. Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 3. Application title
# ============================================================

st.title("📚 PDF RAG Chatbot")

st.markdown(
    """
    Ask questions about the uploaded PDF.

    The chatbot retrieves relevant information from the PDF
    before generating an answer.
    """
)


# ============================================================
# 4. PDF path
# ============================================================

PDF_PATH = os.path.join("data", "sample.pdf")


if not os.path.exists(PDF_PATH):
    st.error(f"PDF not found: {PDF_PATH}")
    st.stop()


# ============================================================
# 5. Build RAG system
# ============================================================

@st.cache_resource
def create_rag_system():

    # -------------------------
    # Load PDF
    # -------------------------
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # -------------------------
    # Split PDF into chunks
    # -------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # -------------------------
    # Load embeddings
    # -------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------
    # Create Chroma database
    # -------------------------
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_rag"
    )

    # -------------------------
    # Create retriever
    # -------------------------
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    # -------------------------
    # Hugging Face client
    # -------------------------
    client = InferenceClient(
        token=HF_TOKEN
    )

    return retriever, client, len(documents), len(chunks)


retriever, client, number_of_pages, number_of_chunks = create_rag_system()


# ============================================================
# 6. RAG function
# ============================================================

def ask_pdf(question):

    # -------------------------
    # Retrieve relevant chunks
    # -------------------------
    results = retriever.invoke(question)

    if not results:
        return "I could not find the answer in the PDF.", []

    # -------------------------
    # Combine context
    # -------------------------
    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    # -------------------------
    # Ask LLM
    # -------------------------
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
You are a helpful PDF question-answering assistant.

Use ONLY the provided PDF context to answer the question.

If the answer cannot be found in the provided context,
say exactly:

"I could not find the answer in the PDF."

Do not use outside knowledge.
Do not invent information.

Give a clear and concise answer.
"""
            },
            {
                "role": "user",
                "content": f"""
PDF CONTEXT:

{context}

QUESTION:

{question}
"""
            }
        ]
    )

    answer = response.choices[0].message.content

    # -------------------------
    # Collect source pages
    # -------------------------
    sources = []

    for doc in results:

        page = doc.metadata.get("page")

        if page is not None:
            sources.append(page + 1)

    # Remove duplicates and sort
    sources = sorted(set(sources))

    return answer, sources


# ============================================================
# 7. Sidebar
# ============================================================

with st.sidebar:

    st.header("📄 PDF Information")

    st.write(f"**File:** `sample.pdf`")

    st.write(
        f"**Pages:** {number_of_pages}"
    )

    st.write(
        f"**Chunks:** {number_of_chunks}"
    )

    st.write(
        "**Embedding model:** "
        "`all-MiniLM-L6-v2`"
    )

    st.write(
        "**Vector database:** Chroma"
    )

    st.write(
        "**LLM:** GPT-OSS-120B"
    )

    st.divider()

    st.info(
        "The chatbot answers using information retrieved from the PDF."
    )


# ============================================================
# 8. Chat history
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 9. Display previous messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "sources" in message and message["sources"]:

            st.caption(
                "Source pages: "
                + ", ".join(
                    str(page)
                    for page in message["sources"]
                )
            )


# ============================================================
# 10. Chat input
# ============================================================

question = st.chat_input(
    "Ask a question about the PDF..."
)


# ============================================================
# 11. Process question
# ============================================================

if question:

    # -------------------------
    # Display user question
    # -------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -------------------------
    # Generate answer
    # -------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching the PDF..."):

            try:

                answer, sources = ask_pdf(question)

                st.markdown(answer)

                if sources:

                    st.caption(
                        "Source pages: "
                        + ", ".join(
                            str(page)
                            for page in sources
                        )
                    )

                else:

                    st.caption("Source pages: Not available")

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )