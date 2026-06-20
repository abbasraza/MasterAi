from pathlib import Path
import logging

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownChunker:
    """
    Splits markdown into chunks based on headers.
    Preserves source metadata on every chunk.
    """

    HEADERS_TO_SPLIT = [
        ("#",   "section"),
        ("##",  "section"),
        ("###", "subsection"),
    ]

    def __init__(self):
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on = self.HEADERS_TO_SPLIT,
            strip_headers       = False,
        )
        self.logger = logging.getLogger(__name__)

    def chunk(self, markdown: str, source_file: str) -> list[Document]:
        """
        Split markdown into chunks.
        Falls back to single chunk if no headers found.
        """
        chunks = self.splitter.split_text(markdown)

        if not chunks:
            self.logger.warning(
                f"No headers found in {source_file} "
                f"— using single chunk"
            )
            chunks = [Document(page_content=markdown)]

        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content = chunk.page_content,
                metadata     = {
                    "doc_type":    "markdown_chunk",
                    "source_file": source_file,
                    "chunk_index": i,
                    "section":     chunk.metadata.get("section", ""),
                    "subsection":  chunk.metadata.get("subsection", ""),
                }
            ))

        self.logger.info(f"{source_file}: {len(docs)} chunks")
        return docs

    def chunk_all(
        self,
        conversions: list[tuple[Path, str]],
    ) -> list[Document]:
        """
        Chunk all (pdf_path, markdown) pairs.
        Returns flat list of all documents.
        """
        all_docs = []
        for pdf_path, markdown in conversions:
            docs = self.chunk(markdown, pdf_path.name)
            all_docs.extend(docs)

        self.logger.info(f"Total chunks: {len(all_docs)}")
        return all_docs