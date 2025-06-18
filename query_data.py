from flask import Flask, request, jsonify
import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_together import Together
from dotenv import load_dotenv
import pickle
import hashlib
import sys
import google.generativeai as genai  # Add this import

# Load environment variables
load_dotenv()

app = Flask(__name__)

CHROMA_PATH = "chroma"
# DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" 

DEFAULT_LLM_MODEL = "gemini-2.5-flash"  # For reference, not used directly now

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Add caching functions to speed up repeated queries
def get_cached_response(prompt, cache_file="response_cache.pkl"):
    """Get cached response if it exists"""
    try:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        if prompt_hash in cache:
            print("Using cached response")
            return cache[prompt_hash]
    except (FileNotFoundError, EOFError):
        pass
    return None

def cache_response(prompt, response, cache_file="response_cache.pkl"):
    """Cache the response for future use"""
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    try:
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    except (FileNotFoundError, EOFError):
        cache = {}
    
    cache[prompt_hash] = response
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

def process_query(query_text, embedding_model=DEFAULT_EMBEDDING_MODEL, llm_model=DEFAULT_LLM_MODEL, use_openai=False, force_reload=False):
    """Process a query and return the response"""
    embedding_function = HuggingFaceEmbeddings(
        model_name=embedding_model,
        cache_folder="./models/"
    )

    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Database not found at {CHROMA_PATH}. Please run create_database.py first.")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    if len(results) == 0 or results[0][1] < 0.1:
        raise ValueError("Unable to find matching results.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    cached_response = get_cached_response(prompt)
    if cached_response:
        response_text = cached_response
    else:
        # Use Gemini instead of Together
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        response_text = response.text
        cache_response(prompt, response_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    return {"response": response_text, "sources": sources}

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get("query_text")
    embedding_model = data.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    llm_model = data.get("llm_model", DEFAULT_LLM_MODEL)
    use_openai = data.get("use_openai", False)
    force_reload = data.get("force_reload", False)

    if not query_text:
        return jsonify({"error": "query_text is required"}), 400

    try:
        result = process_query(query_text, embedding_model, llm_model, use_openai, force_reload)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Check if a query is passed as a command-line argument
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])  # Combine all arguments into a single query string
        try:
            result = process_query(query_text)
            print("Response:", result["response"])
            print("Sources:", result["sources"])
        except Exception as e:
            print("Error:", str(e))
    else:
        # Run the Flask app if no query is provided
        app.run(host="0.0.0.0", port=5000)