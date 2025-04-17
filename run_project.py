"""
One-click script to set up and run the LLM Experimentation project
"""

import os
import sys
import subprocess
import time

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def run_command(command, error_message=None):
    """Run a command and handle errors."""
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        if error_message:
            print(f"ERROR: {error_message}")
        return False

def check_gpu():
    """Check if GPU is available for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU detected: {device_name}")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ No GPU detected. Running on CPU (this will be slower)")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet. Will check GPU after installation.")
        return False

def main():
    """Run the entire project setup and execution."""
    print_header("LLM Experimentation with LlamaIndex - Automated Setup")
    
    # Check for Python
    python_command = "python" if sys.platform == "win32" else "python3"
    
    # Check for GPU
    print("\nChecking for GPU...")
    has_gpu = check_gpu()
    
    # Step 1: Check dependencies
    print("\nChecking and installing dependencies...")
    
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        if not run_command([python_command, "-m", "venv", "venv"],
                         "Failed to create virtual environment"):
            return
    
    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        activate_script = os.path.join("venv", "Scripts", "activate")
        pip_command = os.path.join("venv", "Scripts", "pip")
    else:
        activate_script = os.path.join("venv", "bin", "activate")
        pip_command = os.path.join("venv", "bin", "pip")
    
    # Install PyTorch with CUDA if GPU available
    print("Installing PyTorch...")
    if has_gpu:
        torch_command = f"{pip_command} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
    else:
        torch_command = f"{pip_command} install torch torchvision torchaudio"
    
    if not run_command(torch_command.split(), "Failed to install PyTorch"):
        return
    
    # Install other dependencies
    print("Installing other dependencies...")
    if not run_command([pip_command, "install", "-r", "requirements.txt"],
                     "Failed to install dependencies"):
        return
    
    # Step 2: Check for data
    print_header("Checking for Paul Graham essays")
    
    if not os.path.exists("data") or not os.listdir("data"):
        print("No essays found. Downloading sample essays...")
        if os.path.exists("download_sample_essays.py"):
            if not run_command([python_command, "download_sample_essays.py"],
                             "Failed to download sample essays"):
                print("Please manually download some essays to the 'data' directory before continuing.")
                return
        else:
            print("Download script not found. Please manually add some essays to a 'data' directory.")
            return
    else:
        print(f"Found {len(os.listdir('data'))} essays in the data directory.")
    
    # Step 3: Index the data
    print_header("Indexing Paul Graham essays")
    
    if not os.path.exists("storage") or not os.listdir("storage"):
        print("No index found. Creating index...")
        if not run_command([python_command, "src/index_data.py"],
                         "Failed to index essays"):
            return
    else:
        print("Index found. Skipping indexing step.")
    
    # Step 4: Run the generation script
    print_header("Running Question-Answering System")
    
    try:
        subprocess.run([python_command, "src/generate_text.py"])
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
    
    print_header("Project Execution Completed")
    print("\nTo run again:")
    print(f"1. Activate the virtual environment: {activate_script}")
    print(f"2. Run the generation script: {python_command} src/generate_text.py")
    print("\nTo explore with Jupyter Notebook:")
    print(f"1. Activate the virtual environment: {activate_script}")
    print(f"2. Run Jupyter: {python_command} -m jupyter notebook notebooks/experimentation.ipynb")

if __name__ == "__main__":
    main()
