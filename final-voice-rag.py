import os
import datetime
import ollama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

# Initialize Ollama client
ollama_client = ollama.Client()

model = "llama3.2:3b"

# Ensure data directory exists
if not os.path.exists("./data"):
    os.makedirs("./data")

pdf_files = [f for f in os.listdir("./data") if f.endswith(".pdf")]

if not pdf_files:
    print("No PDF files found in ./data")
    exit()

all_pages = []

for pdf_file in pdf_files:
    file_path = os.path.join("./data", pdf_file)

    print(f"Processing PDF file: {pdf_file}")

    # Load PDF
    loader = PDFPlumberLoader(file_path=file_path)
    pages = loader.load()

    print(f"Pages loaded: {len(pages)}")

    all_pages.extend(pages)

    if pages:
        # Extract text from first page
        text = pages[0].page_content

        print(
            f"Text extracted from the PDF file '{pdf_file}':\n"
            f"{text[:200]}...\n"
        )

        # Prepare summary prompt
        summary_prompt = f"""
You are an AI assistant that helps with summarizing PDF documents.

Here is the content of the PDF file '{pdf_file}':

{text}

Please summarize the content of this document in a few sentences.
"""

        try:
            response = ollama_client.generate(
                model=model,
                prompt=summary_prompt
            )

            summary = response.get("response", "")

            # Uncomment if you want to see the summary
            # print(f"Summary:\n{summary}\n")

        except Exception as e:
            print(
                f"An error occurred while summarizing "
                f"the PDF file '{pdf_file}': {str(e)}"
            )


# ============================================================
# SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300
)

text_chunks = []

for page in all_pages:
    chunks = text_splitter.split_text(page.page_content)
    text_chunks.extend(chunks)

print(f"Number of text chunks: {len(text_chunks)}")


# ============================================================
# CREATE METADATA
# ============================================================

def add_metadata(chunks, doc_title):
    metadata_chunks = []

    for chunk in chunks:
        metadata = {
            "title": doc_title,
            "author": "US Business Bureau",
            "date": str(datetime.date.today()),
        }

        metadata_chunks.append({
            "text": chunk,
            "metadata": metadata
        })

    return metadata_chunks


metadata_text_chunks = add_metadata(
    text_chunks,
    "BOI US FinCEN"
)


# ============================================================
# CREATE DOCUMENT OBJECTS
# ============================================================

docs = [
    Document(
        page_content=chunk["text"],
        metadata=chunk["metadata"]
    )
    for chunk in metadata_text_chunks
]


# ============================================================
# CREATE / LOAD CHROMA VECTOR DATABASE
# ============================================================

from langchain_chroma import Chroma

vector_db_path = "./db/vector_db"
collection_name = "docs-local-rag"

# Ollama embedding model
fastembedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

# If database already exists, load it
if os.path.exists(vector_db_path):

    print("Loading existing vector database...")

    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=fastembedding,
        persist_directory=vector_db_path,
    )

    print("Vector database loaded successfully.")

else:

    print("Creating vector database...")

    # Create empty Chroma database
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=fastembedding,
        persist_directory=vector_db_path,
    )

    # IMPORTANT:
    # Add documents in small batches instead of using
    # Chroma.from_documents() on all documents at once.

    batch_size = 5

    for i in range(0, len(docs), batch_size):

        batch = docs[i:i + batch_size]

        vector_db.add_documents(batch)

        print(
            f"Added "
            f"{min(i + batch_size, len(docs))}"
            f"/{len(docs)} chunks"
        )

    print("Vector database created successfully.")


# ============================================================
# MULTI-QUERY RETRIEVER
# ============================================================

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever


# LLM
local_model = "llama3.2:3b"

llm = ChatOllama(
    model=local_model
)


# Prompt for generating alternative queries
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Generate five different search queries for the user's question.

Keep the meaning of the original question.
Do not introduce unrelated topics or assumptions.

Return only the five alternative queries,
one query per line.

Original question:
{question}
"""
)


retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(
        search_kwargs={"k": 8}
    ),
    llm,
    prompt=QUERY_PROMPT
)


# ============================================================
# RAG PROMPT
# ============================================================

template = """You are answering a question using ONLY the provided context.

Rules:
- Answer directly and clearly.
- Do not use information outside the context.
- Do not mention retrieval, vector databases, or documents.
- If the answer cannot be found in the context, say:
  "I could not find the answer in the provided information."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)


# ============================================================
# RAG CHAIN
# ============================================================

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# ASK QUESTION
# ============================================================

questions = "What is this document about? Explain in detail"

print("\nInvoking chain...\n")

response = chain.invoke(questions)

print("\n==============================")
print("RAG RESPONSE")
print("==============================\n")

print(response)


# ============================================================
# TEXT TO SPEECH WITH ELEVENLABS
# ============================================================

from elevenlabs.client import ElevenLabs
from elevenlabs import stream

text_response = response

api_key = os.getenv("ELEVENLABS_API_KEY")

if not api_key:
    raise ValueError(
        "ELEVENLABS_API_KEY not found in environment variables. "
        "Please check your .env file."
    )

client = ElevenLabs(api_key=api_key)

audio_stream = client.text_to_speech.stream(
    voice_id="cgSgspJ2msm6clMCkdW9",
    text=text_response,
    model_id="eleven_flash_v2_5",
    output_format="mp3_22050_32",
)

voices = client.voices.get_all()

for voice in voices.voices:
    print(f"{voice.name}: {voice.voice_id}")

stream(audio_stream)