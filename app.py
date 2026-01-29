from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

from src.helper import download_embeddings
from src.prompt import prompt

from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

from transformers import pipeline

# ---------------- Flask ----------------
app = Flask(__name__)

# ---------------- Env ----------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ---------------- Embeddings ----------------
embeddings = download_embeddings()
index_name = "medical-chatbot"

# ---------------- Vector store ----------------
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_kwargs={
        "k": 6,
        
    }
)


# ---------------- LLM ----------------
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512,
    temperature=0.2,
    repetition_penalty=1.1,
)


llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ---------------- RAG chain (CORRECT) ----------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | create_stuff_documents_chain(llm=llm, prompt=prompt)
)

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("msg")

    if not msg:
        return jsonify({"answer": "Please enter a question."})

    greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening"]
    if msg.lower().strip() in greetings:
        return jsonify({
            "answer": "Hello! I’m a medical chatbot. You can ask me health-related questions."
        })

    result = rag_chain.invoke(msg)

    if isinstance(result, dict):
        answer = result.get("output_text", str(result))
    else:
        answer = result

    return jsonify({"answer": answer})



# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
