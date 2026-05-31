from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config

# Load API keys from config
config = load_config()

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = InMemoryVectorStore(embeddings)


file_path = "./example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
ids = vector_store.add_documents(documents=all_splits)

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])


# results = vector_store.similarity_search(
#     "How many distribution centers does Nike have in the US?"
# )

# print(results[0])

results = vector_store.similarity_search(
    "How many retail stores does Nike have outside USA?", k=10
)
print(results[0])
