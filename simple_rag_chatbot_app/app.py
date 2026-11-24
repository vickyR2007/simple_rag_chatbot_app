import os
import shutil
import tempfile
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def build_retriever(pdf_path: str):
    """Load a PDF, split it into chunks, and build a FAISS retriever."""
    if not pdf_path:
        raise gr.Error("Please upload a PDF file.")

    temp_dir = tempfile.mkdtemp(prefix="rag_pdf_")
    local_pdf = Path(temp_dir) / Path(pdf_path).name
    shutil.copy2(pdf_path, local_pdf)

    try:
        documents = PyPDFLoader(str(local_pdf)).load()
        if not documents:
            raise gr.Error("No readable text was found in the PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        status = (
            f"PDF indexed successfully. Pages loaded: {len(documents)} | "
            f"Chunks created: {len(chunks)}"
        )
        return retriever, status
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Unable to process the PDF: {exc}") from exc


def ask_question(query: str, retriever):
    """Retrieve relevant PDF chunks and answer using the Groq LLM."""
    if not query or not query.strip():
        return "", ""

    if retriever is None:
        raise gr.Error("Upload and process a PDF before asking a question.")

    if not GROQ_API_KEY:
        raise gr.Error("GROQ_API_KEY is missing. Add it to the .env file and restart the app.")

    docs = retriever.invoke(query.strip())

    context_parts = []
    source_parts = []

    for index, doc in enumerate(docs, start=1):
        page_number = doc.metadata.get("page", "N/A")
        if isinstance(page_number, int):
            page_number += 1
        source = Path(str(doc.metadata.get("source", "Uploaded PDF"))).name

        context_parts.append(
            f"[Source {index} - File: {source}, Page: {page_number}]\n"
            f"{doc.page_content}"
        )
        source_parts.append(
            f"Source {index}\n"
            f"File: {source}\n"
            f"Page: {page_number}\n"
            f"Content:\n{doc.page_content[:500]}\n"
            f"{'-' * 60}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.
Answer the user's question using only the supplied context.
Do not use outside knowledge.
If the answer is not present in the context, reply exactly:
I could not find the answer in the provided document.

Context:
{context}

Question:
{query.strip()}

Answer:
""".strip()

    try:
        llm = ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0.2,
        )
        response = llm.invoke(prompt)
        return response.content, "\n\n".join(source_parts)
    except Exception as exc:
        raise gr.Error(f"Unable to generate the answer: {exc}") from exc


with gr.Blocks(title="PDF RAG Chatbot") as demo:
    gr.Markdown(
        "# 📄 PDF RAG Chatbot\n"
        "Upload a PDF, build a FAISS index, and ask questions grounded in the document."
    )

    retriever_state = gr.State(value=None)

    with gr.Row():
        pdf_input = gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            type="filepath",
        )
        process_button = gr.Button("Process PDF", variant="primary")

    status_output = gr.Textbox(label="Indexing Status", interactive=False)
    question_input = gr.Textbox(
        label="Question",
        placeholder="Ask a question about the uploaded PDF...",
        lines=2,
    )
    ask_button = gr.Button("Ask Question", variant="primary")

    answer_output = gr.Textbox(label="Answer", lines=8)
    sources_output = gr.Textbox(label="Retrieved Sources", lines=15)

    process_button.click(
        fn=build_retriever,
        inputs=pdf_input,
        outputs=[retriever_state, status_output],
    )

    ask_button.click(
        fn=ask_question,
        inputs=[question_input, retriever_state],
        outputs=[answer_output, sources_output],
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, retriever_state],
        outputs=[answer_output, sources_output],
    )


if __name__ == "__main__":
    demo.launch()
