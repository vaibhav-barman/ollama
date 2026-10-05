# 🛒 AI Grocery List Categorizer

A Python utility that leverages a local Large Language Model (**Ollama**) to automatically parse, sort, and organize messy grocery lists into structured categories.

## ✨ Features
* 🧠 **Local AI Processing:** Uses `llama3.2:1b` running locally on your machine—no cloud API keys or internet required.
* 🗂️ **Smart Categorization:** Automatically groups items into Produce, Dairy, Meat, Bakery, Beverages, and more.
* 🔤 **Alphabetical Sorting:** Alphabetizes items within their respective categories.
* 💾 **File Automations:** Reads directly from an input text file and exports the structured results cleanly to a new file.

## 🛠️ Prerequisites

1. Download and install [Ollama](https://ollama.com) for Mac/Windows/Linux.
2. Download the lightweight Llama model via your terminal:
   ```bash
   ollama run llama3.2:1b
   ```

### 🔄 Model Flexibility
While this project defaults to `llama3.2:1b` for speed and efficiency on consumer hardware, you can use any local model supported by Ollama. Simply download your preferred model (e.g., `ollama run llama3:8b` or `ollama run mistral`) and update the `model` variable at the top of `categorizer.py`.

## 🚀 Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd grocery-list-categorizer
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   Create your unorganized list inside `data/grocery_list.txt`. For example:
   ```text
   apples, milk, chicken breast, bananas, bread, eggs, coffee
   ```

4. **Run the script:**
   ```bash
   python categorizer.py
   ```

The script will instantly output the formatted results to your terminal and save them directly inside `data/categorized_grocery_list.txt`.
