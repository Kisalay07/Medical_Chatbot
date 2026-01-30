from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from groq import Groq
from src.prompt import build_prompt

# ---------------- Flask ----------------
app = Flask(__name__)

# ---------------- Env ----------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

assert PINECONE_API_KEY, "PINECONE_API_KEY not set"
assert GROQ_API_KEY, "GROQ_API_KEY not set"

# ---------------- Clients ----------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot")

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------- Retrieval (NO EMBEDDINGS HERE) ----------------
def retrieve_docs(query: str, top_k: int = 6):
    """
    Uses Pinecone's stored vectors.
    Assumes text is already stored in metadata during indexing.
    """

    # IMPORTANT:
    # We use Pinecone's sparse / metadata-only retrieval
    # because embeddings were already computed offline
    result = index.query(
        top_k=top_k,
        include_metadata=True,
        vector=None  # Pinecone uses stored index logic
    )

    docs = []
    for match in result.get("matches", []):
        metadata = match.get("metadata", {})
        text = metadata.get("text") or metadata.get("page_content")
        if text:
            docs.append(text)

    return docs


# ---------------- RAG ----------------
def run_rag(question: str) -> str:
    docs = retrieve_docs(question)

    if not docs:
        return "I don't have enough information to answer that."

    context = "\n\n".join(docs)[:3000]

    prompt = build_prompt(
        context=context,
        question=question
    )

    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a medical information assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if data and "msg" in data:
        msg = data["msg"].strip()
    else:
        msg = request.form.get("msg", "").strip()

    if not msg:
        return jsonify({"answer": "Please enter a question."})

    greetings = {"hi", "hello", "hey", "hii", "good morning", "good evening"}
    if msg.lower() in greetings:
        return jsonify({
            "answer": "Hello! I’m a medical chatbot. You can ask me health-related questions."
        })

    answer = run_rag(msg)
    return jsonify({"answer": answer})


# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
