from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import argparse
import os
from numpy import dot
from numpy.linalg import norm

# Load environment variables
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compare embeddings of words")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                       help=f"HuggingFace model for embeddings (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--word1", type=str, default="apple", 
                       help="First word to compare")
    parser.add_argument("--word2", type=str, default="iphone",
                       help="Second word to compare")
    args = parser.parse_args()
    
    print(f"Using embedding model: {args.embedding_model}")
    
    # Initialize embedding model
    embedding_function = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        cache_folder="./models/"  # Cache the model locally
    )
    
    # Get embedding for a word
    vector = embedding_function.embed_query(args.word1)
    print(f"Vector for '{args.word1}': {vector[:5]}...")  # Show just first 5 elements
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    words = (args.word1, args.word2)
    # Direct comparison method
    vector1 = embedding_function.embed_query(words[0])
    vector2 = embedding_function.embed_query(words[1])
    
    # Calculate cosine similarity manually
    cosine_similarity = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
    
    print(f"Comparing ({words[0]}, {words[1]}): similarity = {cosine_similarity}")
    print(f"Higher value means more similar (1.0 = identical, 0.0 = unrelated)")

if __name__ == "__main__":
    main()
