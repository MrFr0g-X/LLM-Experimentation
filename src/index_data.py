"""
This script indexes Paul Graham's essays using LlamaIndex for efficient retrieval. The indexed data can be used to generate context-aware responses to user queries.
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import sys
import time

def index_essays():
    """
    Indexes Paul Graham's essays by creating a vector store index.
    
    This function reads essay files from the data directory, processes them, and saves the index for later use.
    
    Returns:
        bool: True if indexing is successful, False otherwise.
    """
    
    data_dir = 'data/'
    storage_dir = './storage'
    
    # Check if data directory exists and contains files
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory '{data_dir}' does not exist. Please create it and add Paul Graham's essays.")
        return False
        
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not files:
        print(f"❌ Error: No text files found in '{data_dir}'. Please add Paul Graham's essays as .txt files.")
        return False
    
    print(f"Found {len(files)} essay files in {data_dir}:")
    for file in files:
        file_size = os.path.getsize(os.path.join(data_dir, file)) / 1024
        print(f"- {file} ({file_size:.2f} KB)")
    
    # Create the index
    try:
        print("\nLoading essays from disk...")
        start_time = time.time()
        documents = SimpleDirectoryReader(data_dir).load_data()
        load_time = time.time() - start_time
        print(f"✓ Loaded {len(documents)} documents in {load_time:.2f} seconds")
        
        # Use local HuggingFace embedding model instead of OpenAI
        print("\nInitializing local embedding model (this might take a moment)...")
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✓ Local embedding model initialized")
        
        print("\nCreating index from essays... (this might take a moment)")
        start_time = time.time()
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model
        )
        index_time = time.time() - start_time
        print(f"✓ Index created in {index_time:.2f} seconds")
        
        # Save the index
        print(f"\nSaving index to {storage_dir}...")
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"✓ Index saved successfully to {storage_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        return False

if __name__ == "__main__":
    """
    Entry point for the script. Checks for the required data and initiates the indexing process.
    """
    print("=" * 60)
    print("  Paul Graham Essay Indexer for LlamaIndex")
    print("=" * 60)
    print("This script indexes Paul Graham's essays for efficient retrieval.")
    print("The essays should be placed as .txt files in the 'data/' directory.")
    print("=" * 60 + "\n")
    
    success = index_essays()
    
    if success:
        print("\n" + "=" * 60)
        print("Indexing completed successfully!")
        print("You can now run generate_text.py to query the indexed essays.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Indexing failed. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)
