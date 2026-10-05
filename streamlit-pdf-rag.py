import streamlit as st
import os
import logging
import ollama

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.vectorstores.utils import filter_complex_metadata


# Configure logging
logging.basicConfig(level=logging.INFO)


# Constants
DOC_PATH = "./data/Information Technology Act 2000 - 2008 (amendment).pdf"
MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"
BATCH_SIZE = 5


def ingest_pdf(doc_path):
    """Load PDF document."""

    if os.path.exists(doc_path):
        loader = UnstructuredLoader(file_path=doc_path)
        data = loader.load()

        logging.info("PDF loaded successfully.")
        return data

    logging.error(f"PDF file not found at path: {doc_path}")
    st.error("PDF file not found.")

    return None


def split_documents(documents):
    """Split documents into smaller chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )

    chunks = text_splitter.split_documents(documents)

    # Remove complex metadata that Chroma cannot store
    chunks = filter_complex_metadata(chunks)

    logging.info(f"Documents split into {len(chunks)} chunks.")

    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""

    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Load existing database
    if os.path.exists(PERSIST_DIRECTORY):

        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )

        logging.info("Loaded existing vector database.")

    # Create database for the first time
    else:

        data = ingest_pdf(DOC_PATH)

        if data is None:
            return None

        chunks = split_documents(data)

        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )

        # Add chunks in small batches
        for i in range(0, len(chunks), BATCH_SIZE):

            batch = chunks[i:i + BATCH_SIZE]

            vector_db.add_documents(batch)

            logging.info(
                f"Added {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks."
            )

        logging.info("Vector database created and persisted.")

    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Generate five alternative search queries for the user's question.

Keep the meaning exactly the same as the original question.
Do not introduce new laws, topics, assumptions, or information.

Return only the five queries, one per line.

Original question:
{question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(
            search_kwargs={"k": 10}
        ),
        llm,
        prompt=QUERY_PROMPT
    )

    logging.info("Retriever created.")

    return retriever


def create_chain(retriever, llm):
    """Create the RAG chain."""

    template = """You are answering a question using ONLY the provided context
from the Information Technology Act document.

Instructions:
- Answer the question directly and clearly.
- Do not mention Document objects, document IDs, metadata, retrieval,
  vector databases, or the RAG process.
- Do not use information that is not present in the context.
- If the answer cannot be found in the context, say:
  "I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("RAG chain created successfully.")

    return chain


def main():

    st.title("Information Technology Act Assistant")

    st.write(
        "Ask questions about the Information Technology Act, 2000 "
        "and its 2008 amendment."
    )

    # User input
    user_input = st.text_input(
        "Enter your question:",
        ""
    )

    if user_input:

        with st.spinner("Searching the document..."):

            try:

                # Initialize language model
                llm = ChatOllama(
                    model=MODEL_NAME
                )

                # Load or create vector database
                vector_db = load_vector_db()

                if vector_db is None:
                    st.error(
                        "Failed to load or create the vector database."
                    )
                    return

                # Create retriever
                retriever = create_retriever(
                    vector_db,
                    llm
                )

                # Create RAG chain
                chain = create_chain(
                    retriever,
                    llm
                )

                # Get response
                response = chain.invoke(
                    input=user_input
                )

                st.markdown("### Assistant")
                st.write(response)

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )

    else:

        st.info(
            "Enter a question to get started."
        )


if __name__ == "__main__":
    main()