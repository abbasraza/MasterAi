import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

from chatbot.temporal.activities import (
    activity_parse_pdf,
    activity_validate_report,
    activity_build_documents,
    activity_embed_documents,
)
from chatbot.temporal.workflows import (
    ParsePDFWorkflow,
    ParseAllWorkflow,
    EmbedWorkflow,
    IngestWorkflow,
)

TASK_QUEUE = "medical-rag-queue"
import logging

# This makes workflow.logger actually print to terminal
logging.basicConfig(
    level  = logging.DEBUG,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)

# Specifically enable Temporal workflow logger
logging.getLogger("temporalio.workflow").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Tell Temporal sandbox to NOT restrict these modules
# They use networking/IO which Temporal normally blocks in workflows
# ---------------------------------------------------------------------------

PASSTHROUGH_MODULES = SandboxRestrictions.default.with_passthrough_modules(
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_openai",
    "langchain_text_splitters",
    "docling",
    "chromadb",
    "openai",
    "httpx",
    "requests",
    "urllib3",
    "pydantic",
    "chatbot",          # your own package
    "config",           # your config module
)


async def run_worker():
    print("Connecting to Temporal server...")
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue = TASK_QUEUE,
        workflows  = [
            ParsePDFWorkflow,
            ParseAllWorkflow,
            EmbedWorkflow,
            IngestWorkflow,
        ],
        activities = [
            activity_parse_pdf,
            activity_validate_report,
            activity_build_documents,
            activity_embed_documents,
        ],
        # Apply passthrough restrictions
        workflow_runner=SandboxedWorkflowRunner(
            restrictions=PASSTHROUGH_MODULES
        ),
    )

    print(f"Worker started on queue: {TASK_QUEUE}")
    print("Press Ctrl+C to stop.\n")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())