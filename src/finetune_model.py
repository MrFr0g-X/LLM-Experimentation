"""
This script fine-tunes a GPT-2 model on Paul Graham's essays to improve its ability to generate relevant and insightful responses. The fine-tuning process includes preparing the data, training the model, and saving the fine-tuned model for later use.
"""

import os
import torch
import time
import argparse
import subprocess
import sys
import re
import random
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
    get_scheduler
)

def prepare_data_for_training(data_dir, output_dir):
    """
    Prepares the training data by concatenating essays into a single text file.
    Cleans the content to remove unnecessary markers and formats it for fine-tuning.
    
    Args:
        data_dir (str): Directory containing the essay text files.
        output_dir (str): Directory to save the processed training data.
    
    Returns:
        str: Path to the prepared training data file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_path = os.path.join(output_dir, "train.txt")
    
    with open(train_path, "w", encoding="utf-8") as outfile:
        files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        
        if not files:
            raise ValueError(f"No .txt files found in {data_dir}")
            
        print(f"Processing {len(files)} essay files...")
        
        # First, load all essays to create context-aware examples
        essay_contents = {}
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read().strip()
                
                # Basic cleaning
                content = re.sub(r'\[\d+\]', '', content)  # Remove citation markers
                content = re.sub(r'\nNotes\n+\d+\..*$', '', content, flags=re.DOTALL)  # Remove footnotes
                content = re.sub(r'\n{3,}', '\n\n', content)  # Normalize whitespace
                
                essay_title = filename.replace('.txt', '').replace('_', ' ').title()
                essay_contents[essay_title] = content
        
        # Expanded set of high-quality Q&A examples with consistent formatting
        standard_qa_pairs = [
            {
                "question": "What makes a successful startup according to Paul Graham?",
                "answer": "According to Paul Graham, a successful startup focuses on creating something people genuinely want. The most important factor is rapid growth - successful startups grow at 5-7% per week in their early stages. They start small with a narrow focus, solve real problems for early adopters, and expand gradually. Graham emphasizes that success comes from perseverance, adaptability, and understanding users' needs deeply rather than just pursuing wealth."
            },
            {
                "question": "What does Paul Graham think about programming languages?",
                "answer": "Paul Graham believes programming languages significantly impact productivity and thinking. He favors concise, expressive languages like Lisp that minimize restrictions and maximize power. Graham argues that different languages are suited for different tasks, and the best programmers choose their tools based on appropriateness rather than familiarity. He values languages that allow bottom-up development and rapid prototyping."
            },
            {
                "question": "How does Paul Graham describe the ideal founder?",
                "answer": "According to Paul Graham, the ideal founder is determined, resilient, and able to endure setbacks. They're resourceful problem-solvers who can adapt quickly. The best founders deeply understand their users, possess technical knowledge, and can create what users want. They're comfortable with uncertainty, make decisions quickly with limited information, and are both makers and communicators. Graham emphasizes that persistence matters more than brilliance."
            },
            {
                "question": "What is Paul Graham's philosophy on innovation?",
                "answer": "Paul Graham views innovation as the result of curiosity and a desire to solve genuine problems. He believes that true innovation comes from identifying real needs rather than starting with a solution. Graham suggests that many great innovations are discovered accidentally while working on something else. He emphasizes that innovation requires both technical skill and understanding of user psychology."
            },
            {
                "question": "How should someone get startup ideas according to Paul Graham?",
                "answer": "Paul Graham advises not to try to think of startup ideas deliberately. Instead, he recommends becoming the kind of person who has good ideas organically by working on problems you genuinely care about. Good startup ideas come from noticing problems in your own life, especially those affecting a growing market. He suggests looking for things that seem missing, problems you yourself have, or areas where you have specialized knowledge."
            }
        ]
        
        # Create three different prompt formats to help the model learn flexibility
        prompt_formats = [
            # Format 1: Standard Q&A
            lambda q, a: f"QUESTION: {q}\n\nANSWER: {a}",
            
            # Format 2: According to Paul Graham format
            lambda q, a: f"QUESTION: {q}\n\nAccording to Paul Graham's essays: {a}",
            
            # Format 3: Direct answer format with attribution
            lambda q, a: f"Q: {q}\n\nA: {a}"
        ]
        
        # Write the standard examples with different formats
        print("Adding standard Q&A examples...")
        for qa_pair in standard_qa_pairs:
            # Use each format at least once
            for format_func in prompt_formats:
                formatted_qa = format_func(qa_pair["question"], qa_pair["answer"])
                outfile.write(f"<|startoftext|>\n{formatted_qa}\n<|endoftext|>\n\n")
        
        # Generate context-based examples to help the model learn to use context
        print("Generating context-based examples...")
        context_based_questions = [
            "What does Paul Graham say about %s?",
            "How does Paul Graham approach the topic of %s?",
            "What are Paul Graham's views on %s?",
            "According to Paul Graham, why is %s important?",
            "How should someone approach %s according to Paul Graham?"
        ]
        
        # Topics derived from Paul Graham's essays
        topics = [
            "startups", "innovation", "programming", "wealth creation", 
            "technology", "persistence", "growth", "business strategy",
            "creating value", "ideas", "programming languages", "problem-solving",
            "founders", "venture capital", "great hackers", "scaling startups"
        ]
        
        # Add context-aware examples by extracting sections from essays
        used_topics = set()
        for title, content in essay_contents.items():
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 100]
            if not paragraphs:
                continue
                
            # Extract a meaningful topic from the essay title
            essay_topic = None
            for topic in topics:
                if topic.lower() in title.lower() and topic not in used_topics:
                    essay_topic = topic
                    used_topics.add(topic)
                    break
                    
            if essay_topic is None:
                # Find a topic that appears in the content
                for topic in topics:
                    if topic.lower() in content.lower() and topic not in used_topics:
                        essay_topic = topic
                        used_topics.add(topic)
                        break
                
            # If still no topic, skip this essay
            if essay_topic is None:
                continue
                
            # Choose 2-3 random paragraphs as context
            if len(paragraphs) >= 3:
                context_paragraphs = random.sample(paragraphs, min(3, len(paragraphs)))
                context = "\n\n".join(context_paragraphs)
                
                # Generate a question based on the topic
                question_template = random.choice(context_based_questions)
                question = question_template % essay_topic
                
                # Create answer that references the context
                answer = f"In his essays, Paul Graham discusses {essay_topic} in depth. {context_paragraphs[0][:200]}... He emphasizes the importance of {essay_topic} in the context of startups and innovation."
                
                # Format with context
                context_example = f"According to Paul Graham's essays: {question}\n\n{context}\n\nThe answer is: {answer}"
                outfile.write(f"<|startoftext|>\n{context_example}\n<|endoftext|>\n\n")
        
        # Process and include the original essays
        print("Adding original essay content...")
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read().strip()
                
                # Clean the content
                # Remove citation markers like [1], [2], etc.
                content = re.sub(r'\[\d+\]', '', content)
                
                # Remove footnotes that might appear at the end
                content = re.sub(r'\nNotes\n+\d+\..*$', '', content, flags=re.DOTALL)
                
                # Normalize whitespace
                content = re.sub(r'\n{3,}', '\n\n', content)
                
                # Add special tokens to separate essays
                outfile.write(f"<|startoftext|>\n{content}\n<|endoftext|>\n\n")
        
    print(f"✓ Training data prepared at {train_path}")
    return train_path

def load_dataset(tokenizer, train_path, block_size=128):
    """
    Loads the dataset for training using the specified tokenizer.
    
    Args:
        tokenizer: The tokenizer to preprocess the text data.
        train_path (str): Path to the training data file.
        block_size (int): Maximum sequence length for training.
    
    Returns:
        TextDataset: The prepared dataset for training.
    """
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=block_size  # Context size for training
    )
    return dataset

def check_dependencies():
    """
    Verifies and installs any missing dependencies required for fine-tuning.
    
    Returns:
        bool: True if all dependencies are installed successfully, False otherwise.
    """
    try:
        import accelerate
        print("✓ Accelerate library is already installed")
    except ImportError:
        print("Installing required dependencies for fine-tuning...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.26.0"])
            print("✓ Successfully installed accelerate library")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run manually:")
            print("pip install accelerate>=0.26.0")
            return False
    return True

def fine_tune_model(model_name="gpt2", data_dir="data", output_dir="models/finetuned", 
                    epochs=4, batch_size=4, learning_rate=5e-5, block_size=128):
    """
    Fine-tunes a GPT-2 model on the provided dataset of Paul Graham's essays.
    
    Args:
        model_name (str): Name of the base GPT-2 model to fine-tune.
        data_dir (str): Directory containing the essay text files.
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        block_size (int): Maximum sequence length for training.
    
    Returns:
        str: Path to the directory containing the fine-tuned model.
    """
    # Check dependencies first
    if not check_dependencies():
        return None
        
    print(f"Fine-tuning {model_name} on Paul Graham essays...")
    
    # Create temp directory for processed data
    temp_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Prepare the data
    train_path = prepare_data_for_training(data_dir, temp_dir)
    
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add special tokens for document separation
    special_tokens = {"additional_special_tokens": ["<|startoftext|>", "<|endoftext|>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set pad token for batch processing
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    train_dataset = load_dataset(tokenizer, train_path, block_size)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        logging_dir=os.path.join(output_dir, "logs"),
        prediction_loss_only=True,
        learning_rate=learning_rate,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # We're doing causal language modeling, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Start fine-tuning
    print("Starting fine-tuning process...")
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    print(f"Fine-tuning completed in {training_time:.2f} seconds")
    
    # Save the fine-tuned model
    model_output_dir = os.path.join(output_dir, "paul_graham_gpt2")
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    print(f"Fine-tuned model saved to {model_output_dir}")
    return model_output_dir

if __name__ == "__main__":
    """
    Entry point for the script. Parses command-line arguments and initiates the fine-tuning process.
    """
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on Paul Graham's essays")
    parser.add_argument("--model", type=str, default="gpt2", 
                       choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                       help="Base model to fine-tune")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Directory with essay text files")
    parser.add_argument("--output_dir", type=str, default="models/finetuned",
                       help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=4,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--block_size", type=int, default=128,
                       help="Size of text blocks for training")
    parser.add_argument("--install_deps", action="store_true",
                       help="Install required dependencies before running")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  GPT-2 Fine-tuning on Paul Graham's Essays")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Block size: {args.block_size}")
    print("=" * 60 + "\n")
    
    # Install dependencies if requested
    if args.install_deps:
        try:
            print("Installing required dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers[torch]", "accelerate>=0.26.0"])
            print("✓ Successfully installed required dependencies")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            exit(1)
    
    # Check if the data directory exists and contains files
    if not os.path.exists(args.data_dir) or not os.listdir(args.data_dir):
        print(f"Error: {args.data_dir} directory doesn't exist or is empty.")
        print("Please run download_essays.py first to download Paul Graham's essays.")
        exit(1)
        
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Using GPU for fine-tuning (much faster)")
    else:
        print("⚠️ No GPU detected. Fine-tuning will run on CPU (this will be very slow).")
        print("Consider using a machine with GPU or a cloud GPU service.")
    
    # Run fine-tuning
    model_path = fine_tune_model(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        block_size=args.block_size
    )
    
    print("\n" + "=" * 60)
    print("Fine-tuning completed successfully!")
    print(f"The fine-tuned model is saved at: {model_path}")
    print("You can now use this model for text generation with:")
    print(f"python src/generate_text.py --model {model_path}")
    print("=" * 60)
