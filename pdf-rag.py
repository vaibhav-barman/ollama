# 1) Ingest PDF
from langchain_unstructured import UnstructuredLoader

doc_path = "./data/Information Technology Act 2000 - 2008 (amendment).pdf"
model = "llama3.2:3b"

loader = UnstructuredLoader(file_path=doc_path)
data = loader.load()
print("Done Uploading")


# 2) Split PDF into chunks
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(data)
chunks = filter_complex_metadata(chunks)

print("Done splitting........")


# 3) Create embeddings + save to Chroma
import ollama

ollama.pull("nomic-embed-text")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_db = Chroma(
    collection_name="simple-rag",
    embedding_function=embeddings
)

batch_size = 5

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vector_db.add_documents(batch)
    print(f"Added {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

print("Done adding vector database.....")


# 4) Retrieval
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever

llm = ChatOllama(model=model)

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
    vector_db.as_retriever(search_kwargs={"k": 4}),
    llm,
    prompt=QUERY_PROMPT
)


# 5) RAG prompt
template = """Answer the question based ONLY on the following context.

If the answer is not present in the context, say that you could not find
the answer in the provided document.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)


# 6) RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7) Ask question
question = "What is this Information Technology Act about?"

res = chain.invoke(question)

print("\nAnswer:\n")
print(res)