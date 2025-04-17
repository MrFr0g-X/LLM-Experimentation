"""
Essay Downloader Utility

This script downloads selected Paul Graham essays from his website and prepares them for indexing in the LlamaIndex experiment. It handles content extraction, text normalization, and ensures compatibility with the indexing process.
"""

import os
import requests
from bs4 import BeautifulSoup
import time
import re
import argparse

# List of URLs to some of Paul Graham's popular essays
ESSAY_URLS = [
    "http://paulgraham.com/startupideas.html",   # How to Get Startup Ideas
    "http://paulgraham.com/wealth.html",         # How to Make Wealth
    "http://paulgraham.com/ds.html",             # Do Things that Don't Scale
    "http://paulgraham.com/hp.html",             # Hackers and Painters
    "http://paulgraham.com/gh.html",             # Great Hackers
    "http://paulgraham.com/mean.html",           # Mean People Fail
    "http://paulgraham.com/ace.html",            # What You'll Wish You'd Known
    "http://paulgraham.com/growth.html",         # Startup = Growth
    "http://paulgraham.com/schlep.html",         # Schlep Blindness
    "http://paulgraham.com/ambitious.html"       # Ambitious Startup Ideas
]

def sanitize_filename(title):
    """
    Converts a given title into a valid filename by replacing invalid characters.
    
    Args:
        title (str): The title of the essay.
    
    Returns:
        str: A sanitized filename with a .txt extension.
    """
    # Replace spaces and special chars with underscores
    filename = re.sub(r'[^a-zA-Z0-9]', '_', title).lower()
    # Remove consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    return filename + ".txt"

def extract_content(url):
    """
    Extracts the title and main content from a Paul Graham essay webpage.
    
    Args:
        url (str): The URL of the essay.
    
    Returns:
        tuple: A tuple containing the title and content of the essay.
    """
    try:
        print(f"Fetching {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title - this is simplified and might need adjustment based on the site structure
        title = soup.title.text.replace(" - Paul Graham", "").strip()
        
        # Extract main content - this is simplified and might need adjustment
        # In most of PG's essays, the main content is in <font> tags in the <body>
        content_tags = soup.find_all('font')
        
        # If no specific tags are found, try to get the body content
        if not content_tags:
            content = soup.get_text()
        else:
            # Join the content from all relevant tags
            content = "\n\n".join(tag.get_text() for tag in content_tags)
        
        # Basic cleaning
        content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
        
        return title, content
    
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None, None

def download_essays(output_dir="data", urls=None):
    """
    Downloads a list of Paul Graham essays and saves them as text files.
    
    Args:
        output_dir (str): Directory to save the downloaded essays.
        urls (list): List of essay URLs to download. Defaults to predefined URLs.
    
    Returns:
        None
    """
    if urls is None:
        urls = ESSAY_URLS
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    downloaded = 0
    
    for url in urls:
        title, content = extract_content(url)
        
        if title and content:
            filename = sanitize_filename(title)
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"✓ Downloaded '{title}' to {filepath}")
            downloaded += 1
            
            # Be nice to the website
            time.sleep(1)
    
    print(f"\nFinished! Downloaded {downloaded} essays to {output_dir}")

if __name__ == "__main__":
    """
    Entry point for the script. Parses command-line arguments and initiates the essay download process.
    """
    parser = argparse.ArgumentParser(description='Download Paul Graham essays for the LLM experimentation project.')
    parser.add_argument('--output', '-o', default='data', help='Output directory (default: data)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Paul Graham Essay Downloader")
    print("=" * 60)
    print("This script will download several of Paul Graham's essays")
    print("from paulgraham.com and save them as text files.")
    print("=" * 60 + "\n")
    
    download_essays(args.output)
