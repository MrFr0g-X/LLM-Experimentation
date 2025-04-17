"""
Utility functions for analyzing and managing Paul Graham's essays. Includes functions for environment checks, dataset analysis, and essay summarization.
"""

import os
import torch
import re
from collections import Counter

def check_environment():
    """
    Checks if the required libraries and environment are properly set up.
    
    Returns:
        bool: True if the environment is correctly set up, False otherwise.
    """
    try:
        import transformers
        try:
            import llama_index.core
            print("Required libraries are installed!")
        except ImportError:
            try:
                import llama_index
                print("llama_index is installed, but you should import from llama_index.core")
            except ImportError:
                print("Error: llama_index is not installed. Install with: pip install llama-index")
        
        import pandas
        import numpy
        
        # Check for additional libraries used in the notebook
        try:
            import sklearn
            import seaborn
            import matplotlib
            print("Advanced analysis libraries are also installed!")
        except ImportError:
            print("Note: Some advanced analysis libraries (sklearn, seaborn) might be missing.")
            print("To install: pip install scikit-learn seaborn matplotlib")
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            print("CUDA is available! GPU acceleration can be used.")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU for computations (this might be slower).")
        
        return True
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the missing libraries.")
        return False

def get_dataset_info(data_dir="data/"):
    """
    Retrieves information about the dataset files in the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing dataset files.
    
    Returns:
        list: A list of dataset file names.
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return
    
    files = os.listdir(data_dir)
    if not files:
        print(f"No files found in {data_dir}.")
        return
    
    print(f"Found {len(files)} file(s) in {data_dir}:")
    total_size = 0
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / 1024  # Size in KB
            total_size += size
            print(f"- {file}: {size:.2f} KB")
    
    print(f"Total size: {total_size:.2f} KB")
    return files

def analyze_pg_essay(file_path):
    """
    Analyzes a Paul Graham essay and returns basic statistics such as word count, sentence count, and top words.
    
    Args:
        file_path (str): Path to the essay file.
    
    Returns:
        dict: A dictionary containing analysis results.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Basic stats
        word_count = len(content.split())
        char_count = len(content)
        sentences = content.split('.')
        sentence_count = len(sentences)
        
        # Word frequency
        cleaned_text = re.sub(r'[^\w\s]', '', content.lower())
        words = cleaned_text.split()
        
        # Remove common stopwords
        stopwords = ['the', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'with', 'for', 'is', 'was', 'on']
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Get top words
        top_words = Counter(filtered_words).most_common(10)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'top_words': top_words,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': word_count / sentence_count if sentence_count else 0
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def get_essay_summary(file_path, max_sentences=3):
    """
    Extracts a brief summary from a Paul Graham essay by taking the first few sentences.
    
    Args:
        file_path (str): Path to the essay file.
        max_sentences (int): Maximum number of sentences to include in the summary.
    
    Returns:
        str: A brief summary of the essay.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sentences = content.split('.')
        summary = '.'.join(sentences[:max_sentences]) + '.'
        return summary
    except Exception as e:
        print(f"Error summarizing {file_path}: {e}")
        return ""

if __name__ == "__main__":
    """
    Entry point for testing utility functions. Performs environment checks and analyzes sample essays.
    """
    # Test functionality
    check_environment()
    files = get_dataset_info()
    
    # If there are essay files, analyze the first one as a test
    if files:
        essay_file = os.path.join("data/", files[0])
        print(f"\nAnalyzing essay: {files[0]}")
        stats = analyze_pg_essay(essay_file)
        if stats:
            print(f"Word count: {stats['word_count']}")
            print(f"Sentence count: {stats['sentence_count']}")
            print(f"Average word length: {stats['avg_word_length']:.2f} characters")
            print(f"Average sentence length: {stats['avg_sentence_length']:.2f} words")
            print("Top 10 words:")
            for word, count in stats['top_words']:
                print(f"  {word}: {count}")
            
            print("\nBrief summary:")
            summary = get_essay_summary(essay_file)
            print(summary)
