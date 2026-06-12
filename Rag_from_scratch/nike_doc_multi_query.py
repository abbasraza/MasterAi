from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import os
import sys
from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.load import dumps, loads
from operator import itemgetter

from langchain_groq import ChatGroq
# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


# Load API keys from config
config = load_config()

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = InMemoryVectorStore(embeddings)


file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

vector_store.add_documents(documents=all_splits)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=config['GROQ_API_KEY']
)


# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate two 
different version of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)



generate_queries = (
    prompt_perspectives 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# print("Generated Queries:")
# for query in generate_queries.invoke({"question": "How many retail stores does Nike have outside USA?"}):
#     print(query)
    

retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question": "How many retail stores does Nike have outside USA?"})
print(f"Retrieved {len(docs)} unique documents")
# print(f"Retrieved {len(docs)} unique documents:")
# for doc in docs:
#     print(doc.page_content[:200])
# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question": "How many retail stores does Nike have outside USA?"})
