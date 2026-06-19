# Workflows = orchestration logic (equivalent to @flow in Prefect)
# Workflows are deterministic and durable
# If worker crashes mid-workflow, Temporal replays from last checkpoint

from pathlib import Path
from datetime import timedelta
from typing   import List

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from chatbot.temporal.activities import (
        activity_parse_pdf,
        activity_validate_report,
        activity_build_documents,
        activity_embed_documents,
        ParseInput,
        ParseOutput,
        EmbedInput,
        EmbedOutput,
    )


# ---------------------------------------------------------------------------
# Retry policy (applied to activities)
# ---------------------------------------------------------------------------

RETRY_POLICY = RetryPolicy(
    initial_interval    = timedelta(seconds=2),
    maximum_interval    = timedelta(seconds=30),
    maximum_attempts    = 3,
    non_retryable_error_types=["ValueError"],
)


# ---------------------------------------------------------------------------
# Workflow 1: Parse single PDF
# ---------------------------------------------------------------------------

@workflow.defn(name="ParsePDFWorkflow")
class ParsePDFWorkflow:
    """
    Parse one PDF to JSON.
    Run independently from Temporal UI or CLI.
    """

    @workflow.run
    async def run(self, pdf_path: str, cache_dir: str) -> ParseOutput:
        logger = workflow.logger
        logger.error(f"ParsePDFWorkflow started: {pdf_path}")

        result = await workflow.execute_activity(
            activity_parse_pdf,
            ParseInput(pdf_path=pdf_path, cache_dir=cache_dir),
            start_to_close_timeout = timedelta(minutes=10),
            retry_policy           = RETRY_POLICY,
        )

        valid = await workflow.execute_activity(
            activity_validate_report,
            result,
            start_to_close_timeout = timedelta(seconds=30),
        )

        if not valid:
            logger.warning(f"Validation failed for {pdf_path}")

        return result


# ---------------------------------------------------------------------------
# Workflow 2: Parse all PDFs in a folder
# ---------------------------------------------------------------------------

@workflow.defn(name="ParseAllWorkflow")
class ParseAllWorkflow:
    """
    Parse all PDFs in a folder to JSON.
    Stage 1 — no embeddings, no LLM.
    """

    @workflow.run
    async def run(self, source_path: str, cache_dir: str) -> dict:
        logger    = workflow.logger
        pdf_files = sorted(Path(source_path).rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs")

        results = {}
        skipped = []

        for pdf_file in pdf_files:
            result = await workflow.execute_activity(
                activity_parse_pdf,
                ParseInput(
                    pdf_path  = str(pdf_file),
                    cache_dir = cache_dir,
                ),
                start_to_close_timeout = timedelta(minutes=10),
                retry_policy           = RETRY_POLICY,
            )

            valid = await workflow.execute_activity(
                activity_validate_report,
                result,
                start_to_close_timeout = timedelta(seconds=30),
            )

            if valid:
                results[pdf_file.name] = True
                logger.info(f"Parsed: {pdf_file.name}")
            else:
                skipped.append(pdf_file.name)
                logger.warning(f"Skipped: {pdf_file.name}")

        return {
            "parsed":  len(results),
            "skipped": len(skipped),
            "files":   skipped,
        }


# ---------------------------------------------------------------------------
# Workflow 3: Embed cached JSON into ChromaDB
# ---------------------------------------------------------------------------

@workflow.defn(name="EmbedWorkflow")
class EmbedWorkflow:
    """
    Embed all cached JSON reports into ChromaDB.
    Stage 2 — no LLM.
    """

    @workflow.run
    async def run(
        self,
        cache_dir:   str,
        persist_dir: str,
        force:       bool = False,
    ) -> EmbedOutput:
        logger = workflow.logger
        logger.info(f"EmbedWorkflow started | force={force}")

        result = await workflow.execute_activity(
            activity_embed_documents,
            EmbedInput(
                cache_dir   = cache_dir,
                persist_dir = persist_dir,
                force       = force,
            ),
            start_to_close_timeout = timedelta(minutes=30),
            retry_policy           = RETRY_POLICY,
        )

        logger.info(f"Embed complete: {result.total_documents} documents")
        return result


# ---------------------------------------------------------------------------
# Workflow 4: Full ingestion (parse + embed)
# ---------------------------------------------------------------------------

@workflow.defn(name="IngestWorkflow")
class IngestWorkflow:
    """
    Full ingestion: Parse PDFs + Embed into ChromaDB.
    Stage 1 + 2.
    """

    @workflow.run
    async def run(
        self,
        source_path: str,
        cache_dir:   str,
        persist_dir: str,
        force:       bool = False,
    ) -> dict:
        logger = workflow.logger
        logger.info("IngestWorkflow started")

        # Stage 1: Parse
        parse_result = await workflow.execute_child_workflow(
            ParseAllWorkflow,
            args=[source_path, cache_dir],
            id=f"parse-{workflow.info().workflow_id}",
            task_queue="medical-rag-queue",
        )

        if parse_result["parsed"] == 0:
            logger.error("No PDFs parsed. Aborting embed.")
            return {"status": "failed", "reason": "no PDFs parsed"}

        # Stage 2: Embed
        embed_result = await workflow.execute_child_workflow(
            EmbedWorkflow,
            args=[cache_dir, persist_dir, force],
            id=f"embed-{workflow.info().workflow_id}",
            task_queue="medical-rag-queue",
        )

        return {
            "status":    "success",
            "parsed":    parse_result["parsed"],
            "skipped":   parse_result["skipped"],
            "documents": embed_result.total_documents,
        }