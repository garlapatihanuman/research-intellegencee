"""
Reviewer Agent.

Checks whether the draft answer is supported by the retrieved evidence,
flags unsupported claims or missing information, and approves or rejects
the response. On rejection, the workflow loops back to retrieval (bounded
by MAX_REVIEW_RETRIES to avoid infinite loops).
"""

from __future__ import annotations

from app.agents.llm import chat_json
from app.logging_config import get_logger
from app.models.schemas import ReviewResult, RetrievedItem

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are the Reviewer Agent. You check a draft answer against the
retrieved evidence it was supposedly grounded in.

Respond with ONLY a JSON object of this schema:
{
  "approved": true | false,
  "reason": "short explanation of the decision",
  "missing_information": ["short description", ...]
}

Approve when the draft answer's claims are reasonably supported by the evidence
provided, and appropriately says information was not found where evidence is absent.
Reject when the answer makes claims not backed by any evidence, or when the evidence
appears too thin to answer the question at all.
"""


def review(user_query: str, draft_answer: str, context_items: list[RetrievedItem]) -> ReviewResult:
    """Review a draft answer against retrieved evidence."""
    evidence_summary = "\n".join(
        f"- file={i.filename} page={i.page_number}: {i.text[:200]}" for i in context_items
    ) or "(no evidence retrieved)"

    user_prompt = (
        f"User question:\n{user_query}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        f"Retrieved evidence:\n{evidence_summary}\n\n"
        "Evaluate and respond with the JSON object now."
    )

    data = chat_json(_SYSTEM_PROMPT, user_prompt)
    if not data:
        # Fail safe: approve rather than looping forever on a parsing failure.
        return ReviewResult(approved=True, reason="Review parsing failed; defaulting to approve.")

    result = ReviewResult(
        approved=bool(data.get("approved", True)),
        reason=data.get("reason", ""),
        missing_information=data.get("missing_information", []),
    )
    logger.info("Review result: approved=%s reason=%s", result.approved, result.reason)
    return result
