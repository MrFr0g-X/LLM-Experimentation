"""
Sample Essay Downloader

This script downloads a predefined set of Paul Graham's essays and prepares them for indexing. It ensures proper formatting and saves the essays in the data directory.
"""

import os
import requests
from bs4 import BeautifulSoup
import time
import re

# List of URLs to popular Paul Graham essays
ESSAYS = [
    {
        "url": "http://paulgraham.com/wealth.html",
        "title": "How to Make Wealth"
    },
    {
        "url": "http://paulgraham.com/startupideas.html",
        "title": "How to Get Startup Ideas"
    },
    {
        "url": "http://paulgraham.com/growth.html",
        "title": "Startup Equals Growth"
    },
    {
        "url": "http://paulgraham.com/hp.html",
        "title": "Hackers and Painters"
    },
    {
        "url": "http://paulgraham.com/ds.html",
        "title": "Do Things that Don't Scale"
    }
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
    Extracts the main content from a Paul Graham essay webpage.
    
    Args:
        url (str): The URL of the essay.
    
    Returns:
        str: The extracted content of the essay.
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content - this is simplified
        # In most of PG's essays, the main content is in <font> tags
        content_tags = soup.find_all('font')
        
        if content_tags:
            # Join the content from all relevant tags
            content = "\n\n".join(tag.get_text() for tag in content_tags if len(tag.get_text()) > 100)
        else:
            # Fallback to getting text from the body
            content = soup.body.get_text()
        
        # Basic cleaning
        content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
        
        return content
    
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

def main():
    """
    Downloads a sample set of Paul Graham's essays and saves them in the data directory.
    
    Returns:
        None
    """
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory")
    
    print("Downloading sample Paul Graham essays...")
    
    for essay in ESSAYS:
        url = essay["url"]
        title = essay["title"]
        filename = sanitize_filename(title)
        filepath = os.path.join("data", filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"Essay '{title}' already exists at {filepath}, skipping...")
            continue
        
        content = extract_content(url)
        
        if content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"✓ Downloaded '{title}' to {filepath}")
            # Be nice to the server
            time.sleep(1)
        else:
            print(f"✗ Failed to download '{title}'")
    
    # Check what we've got
    essays = os.listdir("data")
    if essays:
        print(f"\nSuccessfully downloaded {len(essays)} essays to the 'data' directory:")
        for essay in essays:
            size_kb = os.path.getsize(os.path.join("data", essay)) / 1024
            print(f"- {essay} ({size_kb:.1f} KB)")
        
        print("\nNext steps:")
        print("1. Run 'python src/index_data.py' to index the essays")
        print("2. Run 'python src/generate_text.py' to query the essays with GPT-2")
    else:
        print("\nNo essays were downloaded. Please check your internet connection.")

if __name__ == "__main__":
    """
    Entry point for the script. Initiates the sample essay download process.
    """
    main()
