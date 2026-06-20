import argparse
import logging
from pathlib import Path

from chatbot import MedicalRAGChatbot
from chatbot.pipeline import (
    setup_logging,
    flow_parse,
    flow_embed,
    flow_ingest,
    flow_chat,
    flow_run,
)
from chatbot.ingestion.parser import PDFParser


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Medical RAG Chatbot",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # parser.add_argument(
    #     "mode",
    #     choices=["parse", "embed", "ingest", "chat", "run", "debug"],
    #     help=(
    #         "parse   -> Stage 1: PDF to JSON\n"
    #         "embed   -> Stage 2: JSON to ChromaDB\n"
    #         "ingest  -> Stage 1+2: PDF to ChromaDB\n"
    #         "chat    -> Stage 3: Chat\n"
    #         "run     -> All stages\n"
    #         "debug   -> Docling debug\n"
    #     )
    # )

    parser.add_argument(
        "--source",
        default = "./pdfs/",
        help    = "PDF file or folder (default: ./pdfs/)"
    )
    parser.add_argument(
        "--cache-dir",
        default = "./json_cache",
        dest    = "cache_dir",
        help    = "JSON cache dir (default: ./json_cache)"
    )
    parser.add_argument(
        "--persist-dir",
        default = "./chroma_medical_db",
        dest    = "persist_dir",
        help    = "ChromaDB dir (default: ./chroma_medical_db)"
    )
    parser.add_argument(
        "--session-id",
        default = "session_1",
        dest    = "session_id",
        help    = "Chat session ID (default: session_1)"
    )
    parser.add_argument(
        "--force",
        action  = "store_true",
        help    = "Force re-embed even if vectors exist"
    )

    return parser


def main():
    setup_logging()
    args = build_arg_parser().parse_args()

    #Medical rag chatbot initialize
    chatbot = MedicalRAGChatbot(
        source_path = Path(args.source),
        # cache_dir   = Path(args.cache_dir),
        persist_dir = args.persist_dir,
        # session_id  = args.session_id,
    )
    # chatbot.logger.info(f"Mode: {args.mode}")
    chatbot.chat(show_tokens=True)
    
    # if args.mode == "parse":
    #     result = flow_parse(
    #         source_path = Path(args.source),
    #         cache_dir   = Path(args.cache_dir),
    #     )
    #     print(f"\nDone: {result['parsed']} parsed, {result['skipped']} skipped")

    # elif args.mode == "embed":
    #     result = flow_embed(
    #         cache_dir   = Path(args.cache_dir),
    #         persist_dir = args.persist_dir,
    #         force       = args.force,
    #     )
    #     print(f"\nDone: {result['total']} documents indexed")

    # elif args.mode == "ingest":
    #     result = flow_ingest(
    #         source_path = Path(args.source),
    #         cache_dir   = Path(args.cache_dir),
    #         persist_dir = args.persist_dir,
    #         force       = args.force,
    #     )
    #     print(f"\nDone: {result}")

    # elif args.mode == "chat":
    #     flow_chat(
    #         persist_dir = args.persist_dir,
    #         session_id  = args.session_id,
    #     )

    # elif args.mode == "run":
    #     flow_run(
    #         source_path = Path(args.source),
    #         cache_dir   = Path(args.cache_dir),
    #         persist_dir = args.persist_dir,
    #         session_id  = args.session_id,
    #         force       = args.force,
    #     )

    # elif args.mode == "debug":
    #     parser   = PDFParser(cache_dir=Path(args.cache_dir))
    #     pdf_path = Path(args.source)
    #     files    = (
    #         [pdf_path] if pdf_path.is_file()
    #         else sorted(pdf_path.rglob("*.pdf"))
    #     )
    #     for f in files:
    #         parser.debug_document(f)


if __name__ == "__main__":
    main()