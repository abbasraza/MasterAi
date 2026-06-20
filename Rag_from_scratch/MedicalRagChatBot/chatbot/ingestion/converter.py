from pathlib import Path
import logging

from docling.document_converter import DocumentConverter as DoclingConverter


class PDFConverter:
    """
    Converts PDF files to markdown using Docling.
    Zero LLM tokens. Free.
    """

    def __init__(self):
        self.converter = DoclingConverter()
        self.logger    = logging.getLogger(__name__)

    def convert(self, pdf_path: Path) -> str:
        """Convert single PDF to markdown string."""
        self.logger.info(f"Converting: {pdf_path.name}")
        doc      = self.converter.convert(str(pdf_path)).document
        markdown = doc.export_to_markdown()
        self.logger.info(f"Converted: {pdf_path.name} ({len(markdown)} chars)")
        return markdown

    def convert_all(self, source_path: Path) -> list[tuple[Path, str]]:
        """
        Convert all PDFs in a file or directory.
        Returns list of (pdf_path, markdown) tuples.
        """
        pdf_files = self._get_pdf_files(source_path)
        self.logger.info(f"Found {len(pdf_files)} PDFs")

        results = []
        for pdf_file in pdf_files:
            try:
                markdown = self.convert(pdf_file)
                results.append((pdf_file, markdown))
            except Exception as e:
                self.logger.error(f"Failed: {pdf_file.name}: {e}")

        return results

    def _get_pdf_files(self, source_path: Path) -> list[Path]:
        if source_path.is_file():
            if source_path.suffix.lower() != ".pdf":
                raise ValueError(f"Not a PDF: {source_path}")
            return [source_path]
        if source_path.is_dir():
            files = sorted(source_path.rglob("*.pdf"))
            if not files:
                raise ValueError(f"No PDFs in: {source_path}")
            return files
        raise ValueError(f"Path not found: {source_path}")