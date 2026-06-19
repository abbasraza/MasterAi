from pathlib import Path

from langchain_community.callbacks  import get_openai_callback
from langchain_core.output_parsers  import StrOutputParser
from langchain_core.runnables       import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai               import AzureChatOpenAI, AzureOpenAIEmbeddings

from chatbot.ingestion.parser       import PDFParser
from chatbot.ingestion.builder      import DocumentBuilder
from chatbot.retrieval.vector_store import VectorStoreManager
from chatbot.retrieval.retriever    import SmartRetriever
from chatbot.chain.prompt           import MEDICAL_PROMPT
from chatbot.session.session        import SessionManager

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config


# =============================================================================
# Token tracker
# =============================================================================

class TokenTracker:
    """Tracks token usage across all queries in a session."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_tokens      = 0
        self.total_cost        = 0.0
        self.query_count       = 0

    def add(self, cb):
        self.prompt_tokens     += cb.prompt_tokens
        self.completion_tokens += cb.completion_tokens
        self.total_tokens      += cb.total_tokens
        self.total_cost        += cb.total_cost
        self.query_count       += 1

    def query_summary(self, cb) -> str:
        """Summary for a single query."""
        return (
            f"[prompt={cb.prompt_tokens} | "
            f"completion={cb.completion_tokens} | "
            f"total={cb.total_tokens} | "
            f"cost=${cb.total_cost:.6f} | "
            f"session_total={self.total_tokens}]"
        )

    def session_summary(self) -> str:
        """Summary for the full session."""
        return (
            f"\n{'─' * 50}\n"
            f"  Session Token Summary\n"
            f"{'─' * 50}\n"
            f"  Queries          : {self.query_count}\n"
            f"  Prompt tokens    : {self.prompt_tokens}\n"
            f"  Completion tokens: {self.completion_tokens}\n"
            f"  Total tokens     : {self.total_tokens}\n"
            f"  Total cost       : ${self.total_cost:.6f}\n"
            f"{'─' * 50}"
        )


# =============================================================================
# MedicalRAGChatbot
# =============================================================================

class MedicalRAGChatbot:

    def __init__(
        self,
        source_path: Path,
        cache_dir:   Path = Path("./json_cache"),
        persist_dir: str  = "./chroma_medical_db",
        session_id:  str  = "session_1",
    ):
        self.session_id      = session_id
        self.session_manager = SessionManager(max_messages=4)
        self.token_tracker   = TokenTracker()

        config = load_config()

        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint   = config["AZURE_OPENAI_ENDPOINT"],
            api_key          = config["AZURE_OPENAI_API_KEY"],
            azure_deployment = config["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
            api_version      = config["AZURE_OPENAI_EMBEDDINGS_API_VERSION"],
        )

        self.llm = AzureChatOpenAI(
            azure_endpoint   = config["AZURE_OPENAI_ENDPOINT"],
            api_key          = config["AZURE_OPENAI_API_KEY"],
            azure_deployment = config["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            api_version      = config["AZURE_OPENAI_CHAT_API_VERSION"],
            temperature      = 0,
        )

        vs_manager     = VectorStoreManager(embeddings, persist_dir)
        self.retriever = SmartRetriever(vs_manager)

        self._index_if_needed(source_path, cache_dir, vs_manager)
        self.chain = self._build_chain()

    # -------------------------------------------------------------------------

    def _index_if_needed(
        self,
        source_path: Path,
        cache_dir:   Path,
        vs_manager:  VectorStoreManager,
    ):
        if not vs_manager.is_empty():
            print("Reusing existing Chroma store")
            return

        parser    = PDFParser(cache_dir)
        builder   = DocumentBuilder()
        pdf_files = (
            [source_path] if source_path.is_file()
            else sorted(source_path.rglob("*.pdf"))
        )

        all_docs = []
        for pdf_file in pdf_files:
            report = parser.parse(pdf_file)
            docs   = builder.build(report)
            all_docs.extend(docs)
            print(f"  {len(docs)} docs from {pdf_file.name}")

        vs_manager.add(all_docs)
        print(f"Indexed {len(all_docs)} documents total")

    # -------------------------------------------------------------------------

    def _build_chain(self):
        core_chain = (
            {
                "context":      RunnableLambda(
                                    lambda x: self._get_context(x["question"])
                                ),
                "question":     RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
            }
            | MEDICAL_PROMPT
            | self.llm
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            core_chain,
            self.session_manager.get,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    # -------------------------------------------------------------------------

    def _get_context(self, question: str) -> str:
        docs = self.retriever.retrieve(question)
        self.retriever.debug(question, docs)
        return self.retriever.format(docs)

    # -------------------------------------------------------------------------

    def ask(self, question: str) -> tuple[str, dict]:
        """
        Ask a question.
        Returns (answer, token_info)
        """
        self.session_manager.trim(self.session_id)

        with get_openai_callback() as cb:
            answer = self.chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": self.session_id}},
            )

        self.token_tracker.add(cb)

        token_info = {
            "prompt_tokens":     cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens":      cb.total_tokens,
            "cost":              cb.total_cost,
            "session_total":     self.token_tracker.total_tokens,
            "session_cost":      self.token_tracker.total_cost,
        }

        return answer, token_info

    # -------------------------------------------------------------------------

    def chat(self, show_tokens: bool = True):
        """
        Interactive chat loop.
        show_tokens: print token usage after each answer.
        """
        print("\nMedical Records Assistant")
        print("Type exit or quit to stop.")
        if show_tokens:
            print("Token usage shown after each answer.")
        print()

        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not question:
                continue

            if question.lower() in {"exit", "quit"}:
                print(self.token_tracker.session_summary())
                print("Exiting.")
                break

            # Special command: show session summary
            if question.lower() == "tokens":
                print(self.token_tracker.session_summary())
                continue

            # Special command: reset token counter
            if question.lower() == "reset tokens":
                self.token_tracker.reset()
                print("Token counter reset.")
                continue

            answer, token_info = self.ask(question)

            print(f"\nAnswer:\n{answer}\n")

            if show_tokens:
                print(
                    f"[prompt={token_info['prompt_tokens']} | "
                    f"completion={token_info['completion_tokens']} | "
                    f"total={token_info['total_tokens']} | "
                    f"cost=${token_info['cost']:.6f} | "
                    f"session={token_info['session_total']} tokens "
                    f"${token_info['session_cost']:.6f}]"
                )