import argparse
import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import Together  # Updated import for Together AI
from dotenv import load_dotenv
import pickle
import hashlib

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"  # Model ID on Together AI

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


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                       help=f"HuggingFace model for embeddings (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--llm_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                       help="Together AI model ID (default: deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free)")
    parser.add_argument("--use_openai", action="store_true", 
                       help="Use OpenAI instead of Together AI (requires API key)")
    parser.add_argument("--force_reload", action="store_true",
                       help="Force regeneration (ignore cache)")
    
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB with selected embedding model
    print(f"Using embedding model: {args.embedding_model}")
    embedding_function = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        cache_folder="./models/"  # Cache the model locally
    )
    
    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Database not found at {CHROMA_PATH}. Please run create_database.py first.")
        return
        
    # Load the database
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < -9:
            print(f"Unable to find matching results.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Check if we have a cached response
        cached_response = None if args.force_reload else get_cached_response(prompt)
        if cached_response:
            response_text = cached_response
        else:
            # Choose between OpenAI or Together AI
            if args.use_openai:
                from langchain_openai import ChatOpenAI
                print("Using OpenAI model")
                model = ChatOpenAI()
            else:
                # Use Together AI API instead of loading locally
                print(f"Using Together AI model: {args.llm_model}")
                
                # Check if API key is available
                if not os.getenv("TOGETHER_API_KEY"):
                    print("Error: TOGETHER_API_KEY environment variable not set.")
                    print("Please set your Together AI API key in the .env file or as an environment variable.")
                    return
                
                # Initialize the model through Together AI
                model = Together(
                    model=args.llm_model,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9,
                )
            
            # Generate the response
            response_text = model.invoke(prompt)
            # Cache for future use
            cache_response(prompt, response_text)

        # Format and display the results
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
    
    except Exception as e:
        print(f"Error: {e}")
        print("If the error is related to the Together AI API, check your API key and connection.")


if __name__ == "__main__":
    main()