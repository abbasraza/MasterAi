from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings


class VectorStoreManager:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, persist_dir: str = "./chroma_medical_db"):
        self.store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    def is_empty(self) -> bool:
        return self.store._collection.count() == 0

    def add(self, documents: list[Document]):
        self.store.add_documents(documents)

    def as_retriever(self, **kwargs):
        return self.store.as_retriever(**kwargs)