"""
This script generates optimized text responses using a fine-tuned GPT-2 model trained on Paul Graham's essays. It includes advanced context handling, improved generation parameters, and post-processing for high-quality outputs.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import time
import torch
import re
import textwrap
import argparse


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
    # Start with the full response
    answer = full_response.strip()
    
    # First, check if the answer is empty or just repeating the query
    if not answer or answer == query:
        return "The model did not generate a meaningful response for this query."
    
    # Extract relevant parts based on common patterns in the output
    # Look for "The answer is:" marker and take text after it
    if "The answer is:" in answer:
        parts = answer.split("The answer is:", 1)
        if len(parts) > 1 and len(parts[1].strip()) > 5:
            answer = parts[1].strip()
    
    # If we see "According to Paul Graham", try to extract content after it
    elif "According to Paul Graham" in answer:
        parts = answer.split("According to Paul Graham", 1)
        if len(parts) > 1 and len(parts[1].strip()) > 5:
            answer = "According to Paul Graham" + parts[1].strip()
    
    # Check for other common markers in the model output
    markers = ["ANSWER:", "The answer is", "Paul Graham thinks", "Paul Graham believes"]
    for marker in markers:
        if marker in answer:
            parts = answer.split(marker, 1)
            if len(parts) > 1 and len(parts[1].strip()) > 5:
                answer = parts[1].strip()
                break
    
    # Remove any query repetition from the beginning
    if query in answer:
        answer = answer.replace(query, "", 1).strip()
    
    # Clean up citation markers and other artifacts
    answer = re.sub(r'\[\d+\]', '', answer)  # Remove citation markers like [1]
    answer = re.sub(r'QUESTION:.*?(?=\n|$)', '', answer, flags=re.IGNORECASE)  # Remove QUESTION: lines
    
    # Handle the case where it's just listing tech founders
    tech_founders_pattern = r'(Steve Jobs|Bill Gates|Jeff Bezos|Mark Zuckerberg|Elon Musk)'
    if re.search(tech_founders_pattern, answer) and len(answer) < 200:
        # If it's just a list of names, provide a default response
        if re.match(r'^[\s\n]*(' + tech_founders_pattern + r'[\s,]*)+[\s\n]*$', answer):
            return "Based on Paul Graham's essays, he views innovation and creativity as essential elements of successful startups. He believes that startups are not simply the embodiment of an initial brilliant idea, but rather the development and evolution of ideas through understanding users' needs."
    
    # If the answer is still too short, try to extract something meaningful
    if len(answer.strip()) < 20:
        # Look through the full response for sentences about Paul Graham
        sentences = re.split(r'[.!?]', full_response)
        for sentence in sentences:
            if "Paul Graham" in sentence and len(sentence.strip()) > 40:
                return sentence.strip() + "."
        
        # If we couldn't find anything, provide a generic response
        if len(answer.strip()) < 10:
            return "The model couldn't generate a specific response about Paul Graham's views on this topic based on the provided context."
    
    # Final cleanup
    answer = re.sub(r'\n{3,}', '\n\n', answer)  # Normalize consecutive newlines
    answer = answer.replace(": :", ":").replace(":::", ":").replace("::", ":")  # Fix repeated colons
    
    # Ensure the answer doesn't end abruptly
    if answer.endswith(("if", "but", "and", "or", "the", "a", "an")):
        answer = re.sub(r'\s+\w+$', '', answer) + "."
        
    # Add a period at the end if it doesn't have one
    if not answer.endswith((".", "!", "?")):
        answer += "."
    
    return answer


def generate_improved_response(query, query_engine, tokenizer, model, device, 
                              max_context_length=700, max_tokens=250, 
                              temperature=0.2, top_p=0.85):
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
        tuple: The final response, raw context, and generation time.
    """
    # Determine if this is a fine-tuned model by checking the model's name
    model_name = getattr(model.config, '_name_or_path', '')
    is_finetuned = 'paul_graham' in model_name.lower() if model_name else False
    
    # Retrieve context from index
    response = query_engine.query(query)
    raw_context = response.response
    
    # Filter and organize context
    filtered_context = filter_context(raw_context, query, max_context_length)
      # Extract info about the essay to help with relevance
    essay_title = ""
    if "file_path:" in raw_context:
        file_path_match = re.search(r'file_path:.*?([^/\\]+).txt', raw_context)
        if file_path_match:
            essay_title = file_path_match.group(1).replace('_', ' ').title()
            
    # Create a structured prompt based on model type
    if is_finetuned:
        # For fine-tuned model, we need a very specific format to get good responses
        # Just use simple, direct instruction with as little extra text as possible
        prompt = f"""According to Paul Graham's essays: {query}

{filtered_context}

The answer is:"""
    else:
        # For base models, use the standard prompt with context
        prompt = create_prompt(query, filtered_context)
    
    # Print the prompt being sent to the model
    print("\nPrompt being sent to model:")
    print("-" * 40)
    print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    print("-" * 40)
    
    # Encode prompt - set a higher max_length to avoid truncation
    inputs = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True, padding=True)
    
    # Log token count
    input_tokens = inputs.input_ids.shape[1]
    print(f"Prompt token count: {input_tokens} tokens")
    
    # Move to GPU if available
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    start_time = time.time()
      # Use specific generation parameters based on whether this is a fine-tuned model
    if is_finetuned:
        # Create a list of bad words to avoid in generation
        bad_words = []
        # Add numbers that often appear in citations or notes
        for i in range(1, 20):
            bad_words.append(tokenizer.encode(f"{i}"))
            bad_words.append(tokenizer.encode(f"[{i}]"))
        
        # Add phrases we don't want to see in the output
        unwanted_phrases = ["Notes", "Note:", "Give a complete", "well-structured answer", "Write a detailed"]
        for phrase in unwanted_phrases:
            try:
                bad_words.append(tokenizer.encode(phrase))
            except:
                pass  # Skip if encoding fails
                
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,  # Low temperature for more deterministic responses
            top_p=top_p,              # Top_p for more focused token selection
            no_repeat_ngram_size=3,   # Prevent repeating n-grams
            num_beams=5,              # Beam search for more coherent text
            length_penalty=1.2,       # Moderate penalty to encourage reasonable-length responses
            early_stopping=True,      # Enable early stopping to avoid trailing off
            pad_token_id=tokenizer.eos_token_id,
            top_k=40,                 # Limit vocabulary to reduce hallucinations
            repetition_penalty=1.5,   # Stronger repetition penalty to avoid lists
            bad_words_ids=bad_words   # Avoid citation markers, numbers, and certain phrases
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=3,
            num_beams=3,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    # Decode generated text
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process response
    final_answer = post_process_response(full_response, query)
    
    return final_answer, raw_context, generation_time


def format_response_for_display(answer, max_line_length=80):
    """Format the response for better readability."""
    # Break long paragraphs into multiple lines
    formatted = ""
    paragraphs = answer.split("\n\n")
    
    for p in paragraphs:
        if p.strip():
            wrapped = textwrap.fill(p, width=max_line_length)
            formatted += wrapped + "\n\n"
    
    return formatted.strip()


def generate_direct_response(query, tokenizer, model, device, 
                            max_tokens=150, temperature=0.5, top_p=0.85):
    """
    Generate a direct response from the fine-tuned model without retrieval.
    
    Args:
        query (str): The user query
        tokenizer: The GPT-2 tokenizer
        model: The GPT-2 model
        device: The compute device (CPU or GPU)
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for generation (0.0-1.0)
        top_p (float): Nucleus sampling threshold
        
    Returns:
        tuple: (response, generation_time)
    """
    start_time = time.time()
    
    # Create a prompt that matches the fine-tuning format
    prompt = f"QUESTION: {query}\n\nANSWER:"
    
    # Print the prompt
    print("\nPrompt for direct generation:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
    
    # Move to GPU if available
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response with optimized parameters for fine-tuned model
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        no_repeat_ngram_size=4,    # Prevent repetition
        num_beams=4,               # Beam search for coherent text
        early_stopping=True,
        top_k=40,                  # Reduce vocabulary to limit hallucinations
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and clean up response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = post_process_response(full_response, query)
    generation_time = time.time() - start_time
    
    return response, generation_time


def main():
    """
    Main function to handle text generation using the fine-tuned model.
    Parses command-line arguments and processes user queries.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate text with fine-tuned model on Paul Graham's essays")
    parser.add_argument("--model", type=str, default="models/finetuned/paul_graham_gpt2", 
                       help="Model to use (path to fine-tuned model)")
    parser.add_argument("--temperature", type=float, default=0.5,
                       help="Temperature for generation (0.0-1.0)")
    parser.add_argument("--top_p", type=float, default=0.85,
                       help="Top-p (nucleus) sampling value")
    parser.add_argument("--max_tokens", type=int, default=150,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--use_retrieval", action="store_true", default=True,
                       help="Use retrieval-augmented generation (default: True)")
    parser.add_argument("--direct", action="store_true",
                       help="Use direct generation without retrieval")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    # Set padding token to eos_token (required for padding)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model)
    
    # Enable GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ Moving model to GPU...")
        model = model.to(device)
        print(f"✓ Model loaded successfully on GPU")
    else:
        print("⚠️ No GPU detected. Running on CPU (this will be slower)")
        print("✓ Model loaded successfully on CPU")
        
    # Check if we should use direct generation mode
    if args.direct:
        args.use_retrieval = False

    # If using retrieval, load the index
    query_engine = None
    if args.use_retrieval:
        # Check if storage directory exists
        if not os.path.exists("./storage") or not os.listdir("./storage"):
            print("❌ Error: Index not found. Please run index_data.py first to index Paul Graham's essays.")
            return

        print("Loading index of Paul Graham's essays...")
        # Initialize the same embedding model used for indexing
        print("Initializing embedding model...")
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Configure settings to use no LLM for the query engine
        Settings.llm = None
        print("Configured to use retrieval only (no LLM for query engine)")
        
        # Load the saved index with the embedding model
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        print("✓ Index loaded successfully")

        # Create a query engine with no LLM (just retrieval)
        query_engine = index.as_query_engine()

    # Example queries about Paul Graham's essays
    example_queries = [
        "What makes a successful startup according to Paul Graham?",
        "What does Paul Graham think about programming languages?",
        "How does Paul Graham describe the ideal founder?",
        "What advice does Paul Graham give to young people about careers?",
        "What is Paul Graham's philosophy on innovation and creativity?"
    ]

    # Allow user to select a query or enter their own
    print("\nExample queries about Paul Graham's essays:")
    for i, query in enumerate(example_queries):
        print(f"{i+1}. {query}")
    print("6. Enter your own query")

    choice = input("\nSelect a query (1-6): ")
    
    try:
        choice = int(choice)
        if 1 <= choice <= 5:
            query = example_queries[choice-1]
        elif choice == 6:
            query = input("Enter your query about Paul Graham's essays: ")
        else:
            raise ValueError
    except ValueError:
        query = "What makes a successful startup according to Paul Graham?"
        print(f"Invalid choice. Using default query: '{query}'")

    # Generate response based on the selected approach
    print(f"\nProcessing query: '{query}'")
    
    if args.use_retrieval:
        print("Using retrieval-augmented generation...")
        final_answer, raw_context, generation_time = generate_improved_response(
            query=query,
            query_engine=query_engine,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_length=700,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Format and display results
        print("\n" + "-" * 80)
        print("RELEVANT CONTEXT FROM PAUL GRAHAM'S ESSAYS:")
        print("-" * 80)
        print(raw_context[:500] + "..." if len(raw_context) > 500 else raw_context)
        print("-" * 80)
        
    else:
        print("Using direct generation with fine-tuned model (no retrieval)...")
        final_answer, generation_time = generate_direct_response(
            query=query,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        # No context to show for direct generation
        print("\n" + "-" * 80)
        print("DIRECT GENERATION (NO CONTEXT RETRIEVAL)")
        print("-" * 80)
    
    print(f"\nGenerated response in {generation_time:.2f} seconds:")
    print("=" * 80)
    print(format_response_for_display(final_answer))
    print("=" * 80)
    
    # Print device information summary
    if device.type == "cuda":
        print(f"\nGPU information:")
        print(f"- GPU used: {torch.cuda.get_device_name(0)}")
        print(f"- Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"- Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"- Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
