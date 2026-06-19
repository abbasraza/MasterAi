from langchain_core.documents import Document
from chatbot.retrieval.vector_store import VectorStoreManager


class SmartRetriever:

    ABNORMAL_WORDS   = {"abnormal", "high", "low", "flag", "critical", "concern"}
    SUMMARY_WORDS    = {"summary", "all tests", "overview", "full report"}
    DATE_WORDS       = {"when", "date", "performed", "which month", "collected"}
    TREND_WORDS      = {"improve", "better", "worse", "trend", "compare", "progress"}

    def __init__(self, vs_manager: VectorStoreManager):
        self.vs = vs_manager

    def retrieve(self, question: str) -> list[Document]:
        q = question.lower()

        if any(w in q for w in self.ABNORMAL_WORDS):
            return self.vs.as_retriever(
                search_kwargs={"k": 10, "filter": {"is_abnormal": True}}
            ).invoke(question)

        if any(w in q for w in self.SUMMARY_WORDS):
            return self.vs.as_retriever(
                search_kwargs={"k": 5, "filter": {"doc_type": "report_summary"}}
            ).invoke(question)

        if any(w in q for w in self.DATE_WORDS):
            return self.vs.as_retriever(
                search_kwargs={"k": 5, "filter": {"doc_type": "report_summary"}}
            ).invoke(question)

        if any(w in q for w in self.TREND_WORDS):
            return self.vs.as_retriever(
                search_kwargs={"k": 20}
            ).invoke(question)

        return self.vs.as_retriever(
            search_kwargs={"k": 6, "filter": {"doc_type": "individual_test"}}
        ).invoke(question)

    def format(self, documents: list[Document]) -> str:
        parts = []
        for doc in documents:
            source    = doc.metadata.get("source_file",     "unknown")
            rep_date  = doc.metadata.get("report_date",     "?")
            coll_date = doc.metadata.get("collection_date", "?")
            parts.append(
                f"[Source: {source} | Report Date: {rep_date} | Collection Date: {coll_date}]\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    def debug(self, question: str, documents: list[Document]):
        print("\n" + "=" * 80)
        print(f"DEBUG | Question: {question} | Chunks: {len(documents)}")
        print("=" * 80)
        for i, doc in enumerate(documents, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"source={doc.metadata.get('source_file')} | "
                  f"date={doc.metadata.get('report_date')} | "
                  f"type={doc.metadata.get('doc_type')}")
            print(doc.page_content)
        print("=" * 80)