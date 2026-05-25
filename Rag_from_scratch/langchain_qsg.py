import bs4
from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
import sys
from langchain_groq import ChatGroq

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config

# Load API keys from config
config = load_config()

# this is an example of building a RAG system from scratch using langchain. It includes the following steps:
# 1. Load documents from the web using WebBaseLoader
# 2. Split the documents into chunks using RecursiveCharacterTextSplitter
# 3. Embed the chunks using HuggingFaceEmbeddings and store them in a Chroma vectorstore
# 4. Create a retriever from the vectorstore
# 5. Define a prompt template for answering questions based on retrieved documents
# 6. Create a chain that takes a question, retrieves relevant documents, formats them, and passes them
#    to a language model to generate an answer.
# Note: Add your API keys in config_local.py


prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")
#### INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
# prompt = hub.pull("rlm/rag-prompt", dangerously_pull_public_prompt=True)

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=config['GROQ_API_KEY']
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question. this get passed as the "question" variable to the prompt, and the retriever gets invoked with
# the question as well to get the relevant docs from the vectorstore which are passed as "context" to the prompt.
rag_chain.invoke("What is Task Decomposition?")
