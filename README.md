# LLM Experimentation with LlamaIndex

This project explores the capabilities of Large Language Models (LLMs) by leveraging GPT-2 and LlamaIndex to analyze and generate insights from Paul Graham's essays. The project demonstrates how to index essays, retrieve relevant context, and generate meaningful responses to user queries.

## Project Highlights

- **Efficient Essay Indexing**: Uses LlamaIndex to create a searchable index of Paul Graham's essays.
- **Context-Aware Text Generation**: Combines GPT-2 with indexed essays to generate insightful responses.
- **Interactive Querying**: Provides an interactive interface for querying the essays.
- **Customizable Parameters**: Allows experimentation with generation parameters for tailored outputs.

## Example Results

### Query: "What is Paul Graham's advice on startups?"

**Response:**

> "Paul Graham emphasizes the importance of solving real problems and focusing on growth. He advises startups to do things that don't scale initially to build a strong foundation."

### Query: "How does Paul Graham describe hackers?"

**Response:**

> "Hackers are described as creators who enjoy building things and solving problems. They value autonomy and are often driven by curiosity."

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional for faster processing)
- Internet connection for downloading models and dependencies

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MrFr0g-X/LLM-Experimentation.git
   cd LLM-Experimentation
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Verify installation:

   ```bash
   python src/utils.py
   ```

## Usage

### Step 1: Download Essays

Use the provided script to download a set of Paul Graham's essays:

```bash
python download_essays.py
```

Alternatively, manually add essays to the `data/` directory as `.txt` files.

### Step 2: Index the Essays

Create a searchable index of the essays:

```bash
python src/index_data.py
```

### Step 3: Query the Essays

Run the interactive query script:

```bash
python src/generate_text.py
```

### Step 4: Experiment with the Notebook

Explore the essays and generation parameters using the Jupyter notebook:

```bash
jupyter notebook notebooks/experimentation.ipynb
```

## Project Structure

- `data/`: Contains the essay text files.
- `src/`: Core scripts for indexing, querying, and utilities.
- `notebooks/`: Jupyter notebook for experimentation.
- `models/`: Directory for fine-tuned models.
- `storage/`: Stores the indexed essays.

## Future Enhancements

- Expand the dataset to include more essays and related writings.
- Experiment with larger models like GPT-3 or GPT-J.
- Develop a web-based interface for easier querying.
- Implement a citation system to reference specific essays in responses.

