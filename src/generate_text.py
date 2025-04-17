"""
This script provides an interactive interface for generating responses to questions about Paul Graham's essays. It uses a fine-tuned GPT-2 model and a vector store index for context retrieval.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import time
import torch
import re
import textwrap

def filter_context(context, query, max_length=700):
    """
    Filters and organizes the context to make it more relevant to the user's query.
    
    Args:
        context (str): The retrieved context from the index.
        query (str): The user's query.
        max_length (int): Maximum length of the filtered context.
    
    Returns:
        str: The filtered and processed context.
    """
    # Clean up the context
    # Remove file paths and irrelevant markers if present
    cleaned_context = re.sub(r'file_path:.*?\n', '', context)
    cleaned_context = re.sub(r'Context information is below\.[\s\n]*---------------------\n', '', cleaned_context)
    
    # Split context into paragraphs - use larger chunks to maintain coherence
    paragraphs = []
    current_paragraph = ""
    
    for line in cleaned_context.split('\n'):
        line = line.strip()
        if line:
            current_paragraph += line + " "
        elif current_paragraph:  # Empty line and we have content
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
    
    # Add the last paragraph if it exists
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    
    # Extract keywords from query (improved extraction)
    stop_words = {'what', 'does', 'how', 'why', 'when', 'where', 'which', 'paul', 'graham', 
                  'think', 'about', 'according', 'make', 'give', 'describe', 'say', 'tell',
                  'write', 'wrote', 'mentioned', 'discuss', 'explained', 'stated', 'beliefs',
                  'view', 'opinion', 'thought', 'idea', 'concept', 'perspective'}
    
    query_tokens = re.findall(r'\b\w+\b', query.lower())
    query_keywords = set([w for w in query_tokens if len(w) > 2 and w not in stop_words])
    
    # Add important bigrams and trigrams from the query
    important_phrases = []
    for phrase in re.findall(r'\b[\w\s]{5,25}\b', query.lower()):
        words = phrase.split()
        if len(words) >= 2:
            important_phrases.append(phrase)
    
    # Score paragraphs by relevance to query
    scored_paragraphs = []
    for p in paragraphs:
        p_lower = p.lower()
        
        # Base score - keyword matches
        keyword_matches = sum(1 for keyword in query_keywords if keyword in p_lower)
        
        # Phrase matches (weighted higher)
        phrase_matches = sum(2 for phrase in important_phrases if phrase in p_lower)
        
        # Density score (keywords per length)
        density = keyword_matches / (len(p) / 100) if len(p) > 0 else 0
        
        # Combined score
        score = keyword_matches * 1.0 + phrase_matches * 2.0 + density * 0.5
        
        # Boost paragraphs with multiple keyword hits
        if keyword_matches > 1:
            score *= 1.5
        
        scored_paragraphs.append((p, score))
    
    # Sort paragraphs by score (highest first)
    sorted_paragraphs = [p for p, _ in sorted(scored_paragraphs, key=lambda x: x[1], reverse=True)]
    
    # Always include at least one paragraph even if scores are low
    if sorted_paragraphs and all(score == 0 for _, score in scored_paragraphs):
        sorted_paragraphs = paragraphs[:3]  # Take first few paragraphs if no matches
    
    # Take top paragraphs up to max_length
    filtered_context = ""
    current_length = 0
    
    for p in sorted_paragraphs:
        p_length = len(p)
        if current_length + p_length + 2 <= max_length:  # +2 for newlines
            filtered_context += p + "\n\n"
            current_length += p_length + 2
        else:
            # Try to include at least one complete paragraph
            if filtered_context == "" and p_length > max_length:
                filtered_context = p[:max_length-3] + "..."
            break
    
    # If we didn't get enough relevant context, include some of the original
    if current_length < max_length * 0.5 and len(sorted_paragraphs) < len(paragraphs):
        remaining = max_length - current_length
        for p in paragraphs:
            if p not in sorted_paragraphs:
                p_length = len(p)
                if current_length + p_length + 2 <= max_length:
                    filtered_context += p + "\n\n"
                    current_length += p_length + 2
                else:
                    break
    
    return filtered_context.strip()

def create_prompt(query, context):
    """
    Constructs a structured prompt for the model based on the query and context.
    
    Args:
        query (str): The user's query.
        context (str): The filtered context.
    
    Returns:
        str: A well-structured prompt for the model.
    """
    # Simplified prompt that focuses only on the immediate task
    prompt = f"""Answer the following question about Paul Graham's essays using ONLY the information provided below.

CONTEXT FROM PAUL GRAHAM'S ESSAYS:
{context}

QUESTION: {query}

ANSWER:"""
    
    return prompt.strip()

def post_process_response(full_response, query):
    """
    Cleans and formats the model's generated response for better readability.
    
    Args:
        full_response (str): The raw response generated by the model.
        query (str): The original user query.
    
    Returns:
        str: The cleaned and formatted response.
    """
    # Extract only the answer part (after "ANSWER:")
    match = re.search(r'ANSWER:(.*)', full_response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        # If the prompt format wasn't properly followed, try to extract text after the query
        try:
            query_index = full_response.lower().index(query.lower())
            answer = full_response[query_index + len(query):].strip()
        except ValueError:
            answer = full_response
    
    # Clean up citation markers that might have been generated by the model
    answer = re.sub(r'\[\d+\]', '', answer)
    
    # Remove any "Paul Graham" that appears alone at the beginning (common in our fine-tuned output)
    answer = re.sub(r'^Paul\s+Graham\s*\n', '', answer)
    
    # Clean up the answer
    answer = re.sub(r'\n{2,}', '\n\n', answer)  # Normalize multiple newlines
    answer = re.sub(r'[^\w\s.,?!;:()\[\]{}"\'-]', '', answer)  # Remove unusual characters
    
    # If answer is very short, it might be incomplete
    if len(answer) < 20:
        parts = full_response.split('.')
        if len(parts) > 1:
            answer = '.'.join(parts[-3:])  # Take the last few sentences
    
    # Check for and remove incomplete sentences at the end
    sentences = answer.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        answer = '.'.join(sentences[:-1]) + '.'
    
    return answer

def generate_improved_response(query, query_engine, tokenizer, model, device, 
                              max_context_length=700, max_tokens=150, 
                              temperature=0.6, top_p=0.9):
    """
    Generates an improved response using context retrieval and the fine-tuned model.
    
    Args:
        query (str): The user's query.
        query_engine: The query engine for retrieving relevant context.
        tokenizer: The tokenizer for encoding the prompt.
        model: The fine-tuned GPT-2 model.
        device: The compute device (CPU or GPU).
        max_context_length (int): Maximum length for context filtering.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for generation.
        top_p (float): Nucleus sampling threshold.
    
    Returns:
        tuple: The final response, raw context, filtered context, and prompt.
    """
    # Start timing
    start_time = time.time()
    
    # Step 1: Retrieve context from vector store
    raw_response = query_engine.query(query)
    raw_context = str(raw_response)
    
    # Step 2: Filter context to be more relevant to query
    filtered_context = filter_context(raw_context, query, max_length=max_context_length)
    
    # Step 3: Construct prompt with filtered context
    prompt = create_prompt(query, filtered_context)
    
    # Step 4: Generate response with GPT-2
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate with specified parameters
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Step 5: Post-process the response
    final_response = post_process_response(generated_text, query)
    
    # Format the response for display
    formatted_response = textwrap.fill(final_response, width=80)
    
    # Log time taken
    elapsed_time = time.time() - start_time
    print(f"\nGeneration completed in {elapsed_time:.2f} seconds")
    
    return final_response, raw_context, filtered_context, prompt

def load_models_and_index(model_path="models/finetuned/paul_graham_gpt2"):
    """
    Loads the fine-tuned GPT-2 model and the vector store index.
    
    Args:
        model_path (str): Path to the fine-tuned model directory.
    
    Returns:
        tuple: The query engine, tokenizer, model, and compute device.
    """
    print("Loading models and index...")
    start_time = time.time()
    
    # Determine device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set embedding model for LlamaIndex
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Load the fine-tuned GPT-2 model
    print(f"Loading fine-tuned model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    
    # Load the vector index from storage
    print("Loading vector index from storage...")
    if os.path.exists("./storage") and len(os.listdir("./storage")) > 0:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()
        print("Vector index loaded successfully")
    else:
        raise FileNotFoundError("Storage directory doesn't exist or is empty. Please run index_data.py first.")
    
    elapsed_time = time.time() - start_time
    print(f"Models and index loaded in {elapsed_time:.2f} seconds")
    
    return query_engine, tokenizer, model, device

def interactive_session():
    """
    Runs an interactive session where users can ask questions about Paul Graham's essays.
    """
    # Load models and index
    query_engine, tokenizer, model, device = load_models_and_index()
    
    print("\n" + "="*80)
    print("Paul Graham Essay Assistant - Interactive Mode")
    print("="*80)
    print("Ask questions about Paul Graham's essays. Type 'exit' to quit.")
    print("-"*80)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        if not query:
            continue
            
        try:
            # Generate response
            final_response, raw_context, filtered_context, prompt = generate_improved_response(
                query, 
                query_engine, 
                tokenizer, 
                model, 
                device
            )
            
            # Print response
            print("\n" + "-"*80)
            print("RESPONSE:")
            print("-"*80)
            print(textwrap.fill(final_response, width=80))
            print("-"*80)
            
        except Exception as e:
            print(f"\nError generating response: {e}")

if __name__ == "__main__":
    interactive_session()
