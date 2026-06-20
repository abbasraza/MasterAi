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
    def count(self) -> int:
        return self.store._collection.count()

    def add(self, documents: list[Document]):
        self.store.add_documents(documents)

    def as_retriever(self, **kwargs):
        return self.store.as_retriever(**kwargs)
    
    # def retrieve(self, question: str, k: int = 6) -> list[Document]:
    #     """Simple semantic retrieval — no metadata filtering in V1."""
    #     return self.store.as_retriever(
    #         search_kwargs={"k": k}
    #     ).invoke(question)

    # def format(self, documents: list[Document]) -> str:
    #     """Format retrieved docs for LLM context."""
    #     parts = []
    #     for doc in documents:
    #         source  = doc.metadata.get("source_file", "unknown")
    #         section = doc.metadata.get("section",     "")
    #         header  = f"[{source}" + (f" | {section}" if section else "") + "]"
    #         parts.append(f"{header}\n{doc.page_content}")
    #     return "\n\n".join(parts)

    # def debug(self, question: str, documents: list[Document]):
    #     print(f"\n{'='*60}")
    #     print(f"DEBUG | question={question} | chunks={len(documents)}")
    #     print("="*60)
    #     for i, doc in enumerate(documents, 1):
    #         source  = doc.metadata.get("source_file", "?")
    #         section = doc.metadata.get("section",     "?")
    #         print(f"\n--- Chunk {i} ---")
    #         print(f"source={source} | section={section}")
    #         print(doc.page_content[:300])
    #     print("="*60)