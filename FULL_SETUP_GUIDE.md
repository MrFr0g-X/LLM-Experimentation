# Complete Setup and Running Guide for LLM Experimentation Project

This guide will walk you through every step of setting up and running the LLM Experimentation project, with a focus on GPU acceleration.

## Prerequisites

- **Python**: Version 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA and cuDNN**: Properly installed on your system
- **Git**: For cloning the repository (optional)

## Step 1: Setting Up Your Environment

### 1.1 CUDA and GPU Setup

1. **Check your GPU**:
   ```bash
   # On Windows
   nvidia-smi
   
   # On Linux
   nvidia-smi
   ```
   This should display your GPU and CUDA version.

2. **Install CUDA** if not already installed:
   - Download from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
   - Follow their installation instructions for your OS
   - Make sure to install a CUDA version compatible with PyTorch
   
3. **Install cuDNN** (optional but recommended):
   - Download from [NVIDIA's website](https://developer.nvidia.com/cudnn)
   - Follow their installation instructions

### 1.2 Python Environment Setup

1. **Create a virtual environment**:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install PyTorch with CUDA support**:
   ```bash
   # For CUDA 11.7 (replace with your CUDA version)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```

3. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU setup** with PyTorch:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device count: {torch.cuda.device_count()}")
   print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
   ```

## Step 2: Getting The Dataset (Paul Graham's Essays)

For this project, we need to collect several of Paul Graham's essays. You have two options:

### 2.1 Manual Download (More Reliable)

1. Create a `data` folder in your project directory:
   ```bash
   mkdir -p data
   ```

2. Visit [Paul Graham's Essays](http://paulgraham.com/articles.html) website.

3. Select 5-10 essays that interest you, particularly those about startups, programming, or technology. Good choices include:
   - "Hackers and Painters"
   - "How to Start a Startup"
   - "Do Things that Don't Scale"
   - "Startup = Growth"
   - "What You'll Wish You'd Known"

4. For each essay:
   - Open the essay in your browser
   - Select all the content (Ctrl+A)
   - Copy it (Ctrl+C)
   - Create a new text file in the `data` directory with a descriptive name (e.g., `hackers_and_painters.txt`)
   - Paste the content (Ctrl+V)
   - Save the file

### 2.2 Using the Download Script (if available)

If you have the `download_essays.py` script in your project:

```bash
python download_essays.py
```

This will download a predefined set of Paul Graham's essays to your `data` directory.

## Step 3: Indexing the Essays with LlamaIndex

Now that we have the essays, we need to create an index:

```bash
python src/index_data.py
```

This script will:
1. Load the essays from the `data` folder
2. Create vector embeddings for them
3. Store the index in the `storage` directory

You should see output indicating the progress and successful creation of the index.

## Step 4: Running the Question-Answering System

Now you can query Paul Graham's essays:

```bash
python src/generate_text.py
```

This will:
1. Load the GPT-2 model, automatically using your GPU
2. Load the index of Paul Graham's essays
3. Offer you a menu of example queries or let you enter your own
4. Retrieve relevant context from the essays
5. Generate a response based on that context

Since you have GPU acceleration enabled, this process should be faster than running on CPU.

## Step 5: Exploring with the Jupyter Notebook

For more in-depth experimentation:

```bash
jupyter notebook notebooks/experimentation.ipynb
```

The notebook includes:
- Analysis of Paul Graham's essays
- Topic modeling
- Testing different queries
- Experimenting with generation parameters
- Performance comparison of different settings

## Troubleshooting GPU Issues

### CUDA Out of Memory Errors

If you encounter "CUDA out of memory" errors:

1. **Reduce batch size**: Already set to 1 in this project
2. **Free GPU memory**: Close other applications using the GPU
3. **Move to CPU**: If necessary, replace `device = torch.device("cuda")` with `device = torch.device("cpu")`

### Model Not Using GPU

If the model isn't using the GPU even though it's available:

1. **Check CUDA availability**:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. **Check PyTorch CUDA version**:
   ```python
   import torch
   print(torch.version.cuda)
   ```

3. **Ensure model is on GPU**:
   ```python
   print(next(model.parameters()).device)  # Should show 'cuda:0'
   ```

### Slow Performance

If performance is slower than expected with a GPU:
1. Check if your GPU is being fully utilized with `nvidia-smi`
2. Make sure no other applications are using your GPU
3. Update your GPU drivers

## Additional Tips

1. **Monitor GPU usage** during model runs:
   ```bash
   # Keep this running in another terminal while using the model
   watch -n 0.5 nvidia-smi
   ```

2. **Set environment variable** to limit GPU memory usage:
   ```bash
   # Limit GPU memory to 70% of available
   export CUDA_VISIBLE_DEVICES=0  # Use the first GPU
   ```

3. **Mixed precision training** can be enabled for newer GPUs:
   ```python
   # Add this near the top of your script if you have a recent GPU
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

Enjoy experimenting with LlamaIndex and GPT-2 on Paul Graham's essays with the power of your GPU!
