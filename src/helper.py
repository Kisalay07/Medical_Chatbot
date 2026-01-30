import os
from typing import List
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Load PDF files (same name as before)
# -------------------------------------------------
def load_pdf_files(data: str):
    """
    Load PDFs from a directory.
    Returns list of dicts (compatible with trials.ipynb usage).
    """
    documents = []

    for file in os.listdir(data):
        if file.endswith(".pdf"):
            file_path = os.path.join(data, file)
            reader = PdfReader(file_path)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    documents.append(
                        {
                            "page_content": text,
                            "metadata": {"source": file}
                        }
                    )
    return documents


# -------------------------------------------------
# Filter minimal docs (same name, same intent)
# -------------------------------------------------
def filter_to_minimal_docs(docs: List):
    """
    Keep only page_content + source metadata
    """
    minimal_docs = []
    for doc in docs:
        minimal_docs.append(
            {
                "page_content": doc["page_content"],
                "metadata": {"source": doc["metadata"].get("source")}
            }
        )
    return minimal_docs


# -------------------------------------------------
# Text splitter (same function name)
# -------------------------------------------------
def text_split(minimal_docs):
    """
    Split text into chunks (manual replacement for RecursiveCharacterTextSplitter)
    """
    chunk_size = 500
    chunk_overlap = 20

    chunks = []

    for doc in minimal_docs:
        text = doc["page_content"]
        source = doc["metadata"]["source"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                {
                    "page_content": chunk_text,
                    "metadata": {"source": source}
                }
            )
            start = end - chunk_overlap

    return chunks


# -------------------------------------------------
# Download embeddings (same function name)
# -------------------------------------------------
def download_embeddings():
    """
    Return SentenceTransformer model
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)
