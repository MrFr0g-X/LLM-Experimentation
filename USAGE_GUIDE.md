# Comprehensive Usage Guide: LLM Experimentation with LlamaIndex

This guide provides detailed instructions on how to set up, run, and use this project that explores Paul Graham's essays using GPT-2 and LlamaIndex.

## Prerequisites

- **Python**: Version 3.8 or higher
- **Git**: For cloning the repository
- **Memory**: At least 8GB RAM recommended
- **Disk space**: ~2GB for Python, libraries, and model weights
- **Internet connection**: Required for downloading the GPT-2 model on first run

## Step 1: Set Up the Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MrFr0g-X/LLM-Experimentation.git
   cd LLM-Experimentation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python src/utils.py
   ```
   You should see a message confirming the required libraries are installed.

## Step 2: Obtain Paul Graham's Essays

You need to obtain Paul Graham's essays as the dataset for this project. You have two options:

### Option 1: Manual Download (Recommended)

1. Visit [Paul Graham's website](http://paulgraham.com/articles.html)
2. Choose 5-10 essays that interest you (e.g., "Hackers and Painters", "How to Start a Startup")
3. For each essay:
   - Open the essay in your browser
   - Copy the main text content (excluding navigation elements)
   - Create a new .txt file in the `data/` folder with a descriptive filename (e.g., `hackers_and_painters.txt`)
   - Paste the content and save the file

### Option 2: Using the Download Script

We've included a helper script to download some popular essays:

```bash
# Create the data directory if it doesn't exist
mkdir -p data

# Run the download script
python download_essays.py
```

**Note**: Please be respectful of Paul Graham's website and don't scrape aggressively. The script includes a delay between requests.

## Step 3: Index the Essays

Before you can query the essays, you need to index them with LlamaIndex:

```bash
python src/index_data.py
```

This will:
1. Load the essays from the `data/` folder
2. Create a vector index
3. Save the index to the `storage/` directory

You should see output showing the essays being loaded and indexed, with a success message at the end.

## Step 4: Generate Text Based on Queries

Now you can query the indexed essays and generate responses:

```bash
python src/generate_text.py
```

This interactive script will:
1. Load the GPT-2 model
2. Load the index created in Step 3
3. Present you with example queries about Paul Graham's essays
4. Allow you to choose a query or enter your own
5. Generate a response using GPT-2 based on relevant context from the indexed essays

## Step 5: Experiment with the Jupyter Notebook

For more in-depth exploration, use the Jupyter notebook:

```bash
jupyter notebook notebooks/experimentation.ipynb
```

The notebook includes:
- Data visualization of Paul Graham's essays
- Topic analysis
- Multiple query tests
- Parameter experimentation
- Performance analysis

## Common Issues and Troubleshooting

### Issue: "No module named 'X'" error
**Solution**: Install the missing package with `pip install X`

### Issue: "Directory 'data/' does not exist" error
**Solution**: Create the directory and add essay files:
```bash
mkdir data
# Then add .txt files as described in Step 2
```

### Issue: CUDA/GPU errors
**Solution**: The code will fall back to CPU if CUDA is not available. For GPU acceleration, ensure you have the correct CUDA version installed for your PyTorch version.

### Issue: "Index not found" error
**Solution**: Run `python src/index_data.py` first to create the index.

### Issue: Poor or irrelevant responses
**Solution**: 
- Check that your essays are properly formatted text files
- Try more specific queries
- Experiment with generation parameters in the notebook

## Advanced Usage

### Customizing Generation Parameters

In `src/generate_text.py`, you can modify the parameters passed to `model.generate()`:
- `max_new_tokens`: Controls response length
- `temperature`: Higher (>1.0) = more creative, Lower (<0.7) = more focused
- `top_k` and `top_p`: Control diversity of responses

### Adding More Essays

You can add more essays at any time by placing them in the `data/` directory, but you'll need to re-run the indexing:

```bash
python src/index_data.py
```

### Using Different Models

Advanced users can modify `src/generate_text.py` to use different pre-trained models from Hugging Face:

```python
# Example: Using GPT-2 Medium instead of the base model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
```

## Project Structure Reference

