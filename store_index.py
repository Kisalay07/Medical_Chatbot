from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings,
)

from pinecone import Pinecone, ServerlessSpec

# ---------------- Load env ----------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
assert PINECONE_API_KEY, "PINECONE_API_KEY not set"

# ---------------- Load & preprocess PDFs ----------------
extracted_data = load_pdf_files(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# ---------------- Embeddings ----------------
embedding_model = download_embeddings()
DIMENSION = embedding_model.get_sentence_embedding_dimension()

# ---------------- Pinecone ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

index = pc.Index(index_name)

# ---------------- Upload vectors ----------------
vectors = []

for i, doc in enumerate(text_chunks):
    embedding = embedding_model.encode(doc["page_content"]).tolist()

    vectors.append(
        {
            "id": f"doc-{i}",
            "values": embedding,
            "metadata": {
                "text": doc["page_content"],
                "source": doc["metadata"]["source"],
            },
        }
    )

# Upsert in batches (safe)
BATCH_SIZE = 100
for i in range(0, len(vectors), BATCH_SIZE):
    index.upsert(vectors=vectors[i:i + BATCH_SIZE])

print(f"✅ Indexed {len(vectors)} chunks into Pinecone index '{index_name}'")
