# Ollama AI & RAG Projects

A hands-on experimentation and learning repository focused on building foundational AI/LLM systems locally. This workspace documents practical implementations of local large language model inference, retrieval-augmented generation (RAG), vector embeddings, multi-query retrieval, structured function calling, agentic workflows, and external voice AI integration.

The primary objective is to deconstruct modern LLM applications by implementing their core architectural patterns from scratch in a local environment while selectively integrating hosted services where suitable.

---

## 🚀 Projects

### 1. PDF RAG Assistant (`pdf-rag.py` / `streamlit-pdf-rag.py`)

A retrieval-augmented generation pipeline that ingests PDF documents, processes them into semantic embeddings, persists them to a local vector store, and synthesizes answers using a local language model with multi-query retrieval expansion.

```text
PDF Document
     ↓
PDF Loader (PDFPlumber)
     ↓
Recursive Text Chunking
     ↓
Vector Embeddings (nomic-embed-text)
     ↓
Local Vector Store (ChromaDB)
     ↓
MultiQueryRetriever
     ↓
Local LLM Synthesis (Llama 3.2 3B via Ollama)
     ↓
Contextual Answer

```

* **Document Ingestion & Chunking:** Loads source documents and fragments raw text into context-preserving windows with configured chunk sizes and overlap.
* **Embedding & Storage:** Generates dense semantic representations via `nomic-embed-text` and persists vectors to disk using ChromaDB to prevent re-indexing across runs.
* **Multi-Query Retrieval:** Deconstructs user prompts into multiple complementary query perspectives to overcome semantic mismatches between raw prompts and source chunks.
* **Inference:** Feeds retrieved context passages into `llama3.2:3b` executed entirely on local hardware.
* **Interactive UI:** Optional Streamlit web interface (`streamlit-pdf-rag.py`) for document upload and conversational Q&A.

**Technologies:** Python, Ollama, LangChain, ChromaDB, PDFPlumber, Streamlit.

---

### 2. Voice-Enabled PDF RAG Assistant (`final-voice-rag.py`)

Extends the local PDF RAG architecture with a voice synthesis output layer, coupling private local document processing and inference with low-latency external speech generation.

```text
PDF Document
     ↓
PDF Ingestion & Chunking
     ↓
Nomic Embeddings
     ↓
Persistent ChromaDB
     ↓
Multi-Query Semantic Retrieval
     ↓
Local Llama 3.2 Synthesis
     ↓
Generated Answer Text
     ↓
ElevenLabs Streaming TTS (Cloud API)
     ↓
Local Audio Playback (mpv)

```

* **Local Inference & Vector Search:** PDF ingestion, semantic indexing, vector retrieval, and LLM text generation operate completely locally.
* **Voice Synthesis:** The final synthesized response text is streamed to ElevenLabs (`eleven_flash_v2_5`) via an external API call for real-time text-to-speech conversion.
* **Local Playback:** Streams received audio directly to standard output or a local media player (`mpv`) without requiring intermediate disk writes.

> *Note:* While LLM inference and vector storage run strictly on-device, ElevenLabs is a hosted cloud API requiring an active network connection and API credentials.

---

### 3. Grocery Function Calling (`function-calling.py`)

An implementation of tool calling that grants the local LLM the ability to interact with executable Python code, parse structured tool requests, and incorporate real-world execution outputs into conversational responses.

```text
User Request
     ↓
LLM (Ollama)
     ↓
Tool Call Request (JSON Schema)
     ↓
Python Function Execution
     ↓
Tool Result Payload
     ↓
LLM Follow-Up Context
     ↓
Final User-Facing Response

```

* **Tool Definitions & JSON Schemas:** Declares typed function signatures with formal parameter descriptions exposed to the LLM.
* **Target Functions:**
* `fetch_price_and_nutrition`: Retrieves item-level pricing, macro information, and inventory data.
* `fetch_recipe`: Retrieves ingredients and preparation steps based on supplied recipe queries.


* **Execution Loop:** Parses structured tool invocations from the model output, invokes the matching Python asynchronous routines, and feeds execution results back into the context window for final answer formulation.

---

### 4. ReAct Agent Experiments

Exploratory implementations examining reasoning and acting loops (ReAct pattern) using local model inference.

```text
Thought → Action → Observation → Thought → Final Answer

```

These experiments evaluate the reasoning limits and tool-selection reliability of compact local models (e.g., Llama 3.2 3B) when operating over iterative multi-turn planning cycles without fine-tuning.

---

### 5. Local LLM Baseline Experiments (`start-1.py`, `start-2.py`)

Foundational baseline scripts evaluating basic Ollama API connectivity, raw prompt-completion workflows, system prompt steering, and streaming output handling before composing higher-order pipelines.

---

## 🧠 Models Used

All models run locally via Ollama:

| Model | Purpose | Parameters |
| --- | --- | --- |
| `llama3.2:3b` | Primary text inference, synthesis, tool selection | 3 Billion |
| `nomic-embed-text` | High-dimensional text embeddings for ChromaDB vector search | 137 Million |

Pull required models before running the code:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text

```

---

## 🛠️ Tech Stack

| Technology | Purpose |
| --- | --- |
| **Python** | Primary development language (asyncio & typing) |
| **Ollama** | Local LLM and embedding model runtime |
| **Llama 3.2 3B** | Compact local reasoning and text synthesis model |
| **Nomic Embed Text** | Local vector embedding generation |
| **LangChain** | Pipeline composition, chunking, and multi-query retrieval |
| **ChromaDB** | Local persistent vector database for document embeddings |
| **PDFPlumber** | Document extraction and PDF text parsing |
| **ElevenLabs API** | Hosted neural text-to-speech synthesis (streaming) |
| **mpv** | Lightweight command-line audio player for streaming playback |
| **Streamlit** | Rapid web interface for PDF RAG interaction |
| **python-dotenv** | Environment variable management for API keys |

---

## 📁 Project Structure

```text
ollama/
├── data/                    # Source PDF documents for indexing
├── db/                      # Local persistent ChromaDB storage (not committed)
├── function-calling.py      # Async tool definition & function-calling loop
├── final-voice-rag.py       # Local RAG pipeline coupled with ElevenLabs voice output
├── pdf-rag.py               # Core local PDF retrieval-augmented generation script
├── streamlit-pdf-rag.py     # Streamlit web interface for the RAG assistant
├── start-1.py               # Baseline local inference test script
├── start-2.py               # Extended prompt-completion test script
├── Modelfile                # Custom Ollama model configuration (if configured)
├── requirements.txt         # Project dependencies
├── .env.example             # Template for API credentials
├── .gitignore               # Excludes virtualenvs, .env, and local db/
└── README.md

```

> *Note:* The `db/` directory contains locally generated ChromaDB vector files and is excluded via `.gitignore`. Not all files in this workspace represent standalone products; many serve as intermediate experiments.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/vaibhav-barman/ollama.git
cd ollama

```

### 2. Configure Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 4. Install and start Ollama

Ensure Ollama is installed and running on your system, then pull the necessary models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text

```

---

## 🔐 Environment Variables

Scripts utilizing ElevenLabs require an API key. Create a `.env` file in the project root:

```bash
cp .env.example .env

```

Define your credentials inside `.env`:

```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

```

Ensure `.env` remains in `.gitignore` to prevent committing secrets.

---

## 🔊 Voice Output Setup

The voice RAG workflow (`final-voice-rag.py`) uses ElevenLabs for cloud-based voice synthesis and `mpv` for streaming local audio playback.

### Install mpv (macOS)

```bash
brew install mpv
mpv --version

```

### Synthesis Configuration

Audio streaming uses the ElevenLabs client configured with a designated voice and low-latency model:

```python
audio_stream = client.text_to_speech.stream(
    voice_id="YOUR_VOICE_ID",
    text=text_response,
    model_id="eleven_flash_v2_5",
    output_format="mp3_22050_32",
)

```

---

## ▶️ Running the Projects

Ensure your virtual environment is active (`source venv/bin/activate`) and Ollama is running.

**Standard PDF RAG Assistant:**

```bash
python3 pdf-rag.py

```

**Streamlit Web Interface for RAG:**

```bash
streamlit run streamlit-pdf-rag.py

```

**Voice-Enabled PDF RAG:**

```bash
python3 final-voice-rag.py

```

**Grocery Function Calling:**

```bash
python3 function-calling.py

```

**Baseline Inference Tests:**

```bash
python3 start-1.py
python3 start-2.py

```

---

## 🔍 RAG Pipeline Architecture

```text
Input Document (.pdf)
       ↓
[ Document Ingestion ]  → Extracts raw text using PDFPlumber
       ↓
[ Text Chunking ]       → Splits text into structured segments with overlap
       ↓
[ Vector Embedding ]    → Converts chunks into dense vectors via nomic-embed-text
       ↓
[ Persistent Storage ]  → Writes indexed vectors to disk via ChromaDB
       ↓
[ Query Expansion ]     → MultiQueryRetriever generates diverse sub-queries
       ↓
[ Vector Search ]       → Cosine/L2 similarity search across stored chunks
       ↓
[ Context Assembly ]    → Concatenates retrieved passages with system prompts
       ↓
[ LLM Generation ]      → Llama 3.2 synthesizes final answer grounded in context

```

* **Chunking Strategy:** Balances granular semantic resolution against context window overhead.
* **Persistence:** Local ChromaDB eliminates recurrent vector generation overhead across application lifecycles.
* **Query Optimization:** Multi-query expansion mitigates vocabulary mismatch between user inquiries and indexed documentation.

---

## 🧩 Function Calling Pipeline

```text
                User Query
                    │
                    ▼
          [ Local LLM Ingestion ]
                    │
   Does prompt require external data?
       ┌────────────┴────────────┐
       ▼                         ▼
     [ No ]                    [ Yes ]
       │                         │
Formulate direct answer          Emit structured JSON tool arguments
       │                         │
       │                         ▼
       │               [ Execute Python Tool ]
       │               - fetch_price_and_nutrition
       │               - fetch_recipe
       │                         │
       │                         ▼
       │               Receive structured return data
       │                         │
       └────────────┬────────────┘
                    ▼
          [ Final Answer Synthesis ]

```

Function calling decouples reasoning from data ownership: the LLM identifies *when* an operational gap exists, selects an available tool schema, generates valid arguments, and lets Python perform deterministic data retrieval before formatting the result.

---

## 🎓 What I Learned

### Local LLMs & Inference

* Running compact parameter models (`llama3.2:3b`) locally using Ollama.
* Managing context window limitations and latency trade-offs on consumer hardware.
* System prompt tuning to enforce structured reasoning and output fidelity.

### Retrieval-Augmented Generation (RAG)

* Ingesting, parsing, and cleaning raw PDF data structures.
* Tuning text chunking boundaries and overlap parameters.
* Producing, indexing, and querying semantic embeddings with `nomic-embed-text`.
* Managing local persistence layers using ChromaDB.
* Enhancing recall with LangChain's `MultiQueryRetriever`.

### Function Calling & Tools

* Defining machine-readable JSON schemas for Python functions.
* Validating model-produced parameter sets.
* Constructing asynchronous execution loops returning typed results to the LLM.

### Voice Systems

* Interfacing with external streaming speech engines (ElevenLabs).
* Directing network audio byte streams directly into local subprocess media players (`mpv`).

---

## 🎯 Repository Purpose

The goal is to understand how modern LLM applications are constructed by building the core AI pipeline locally, while also experimenting with external AI services where appropriate.

```text
Local LLMs
    ↓
Prompt Engineering
    ↓
Embeddings
    ↓
Vector Databases
    ↓
RAG
    ↓
Multi-Query Retrieval
    ↓
Function Calling
    ↓
Agents
    ↓
Voice AI

```

---

## 🚧 Future Experiments

* [ ] Metadata filtering and dynamic chunk routing
* [ ] Hybrid search implementation (combining BM25 lexical scoring with dense vector search)
* [ ] Cross-encoder reranking layers (e.g., Cohere/BGE reranker)
* [ ] Strict citation generation and automated hallucination checks
* [ ] Streaming tokens directly into voice synthesis pipelines
* [ ] Fully local speech-to-text (Whisper.cpp) for bidirectional voice loops
* [ ] RAG evaluation suites using RAGAS and TruLens
* [ ] Containerized deployment configurations (Docker, Ollama container orchestration)

---

## 👨‍💻 Author

**Vaibhav Barman**

GitHub: [https://github.com/vaibhav-barman](https://github.com/vaibhav-barman)