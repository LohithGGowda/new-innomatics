"""
HITL Handler
------------
Manages the Human-in-the-Loop escalation flow.

When the router flags a query for escalation, the CLI runner calls
`handle_escalation()`. This function:
  1. Prints a clear escalation notice to the terminal
  2. Displays the original query and the reason for escalation
  3. Blocks and waits for a human agent to type a response
  4. Returns the human's answer so it can be stored in GraphState

The handler is intentionally decoupled from LangGraph internals so it can
be replaced with a web-based or ticketing-system integration later.
"""

from __future__ import annotations

import logging
import signal
import sys
from typing import Optional

from rag_assistant.config import config

logger = logging.getLogger(__name__)

# ANSI colour codes for terminal output
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _timeout_handler(signum, frame):
    raise TimeoutError("HITL response timed out.")


def handle_escalation(
    query: str,
    escalation_reason: str,
    llm_draft: Optional[str] = None,
    timeout_seconds: int = config.hitl_timeout_seconds,
) -> str:
    """
    Block and wait for a human agent to provide an answer.

    Args:
        query: The original user question.
        escalation_reason: Why the system escalated this query.
        llm_draft: Optional LLM-generated draft the human can use as a starting point.
        timeout_seconds: Seconds to wait before raising TimeoutError.

    Returns:
        The human agent's response string.

    Raises:
        TimeoutError: If no response is received within timeout_seconds.
        KeyboardInterrupt: If the agent presses Ctrl+C.
    """
    _print_escalation_banner(query, escalation_reason, llm_draft)

    # Set up timeout on POSIX systems (Linux/macOS)
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        print(f"\n{_BOLD}{_CYAN}Human Agent Response{_RESET} (type your answer and press Enter):")
        print(f"{_CYAN}> {_RESET}", end="", flush=True)
        human_response = input().strip()

        if not human_response:
            human_response = (
                "A human agent has been notified and will follow up with you shortly."
            )
            logger.warning("Human agent submitted empty response — using default message.")

    except TimeoutError:
        logger.error("HITL timed out after %d seconds.", timeout_seconds)
        human_response = (
            "We were unable to reach a human agent in time. "
            "Please contact support directly."
        )
    except KeyboardInterrupt:
        logger.warning("HITL interrupted by operator.")
        human_response = "Support session interrupted. Please try again later."
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)  # Cancel the alarm

    print(f"\n{_GREEN}✓ Human response recorded.{_RESET}\n")
    logger.info("HITL response captured (length=%d chars).", len(human_response))
    return human_response


def _print_escalation_banner(
    query: str,
    reason: str,
    llm_draft: Optional[str],
) -> None:
    """Print a formatted escalation notice to stdout."""
    separator = "─" * 60
    print(f"\n{_RED}{_BOLD}{'═' * 60}{_RESET}")
    print(f"{_RED}{_BOLD}  ⚠  HUMAN ESCALATION REQUIRED{_RESET}")
    print(f"{_RED}{_BOLD}{'═' * 60}{_RESET}")
    print(f"\n{_YELLOW}Reason:{_RESET}  {reason}")
    print(f"\n{_YELLOW}User Query:{_RESET}")
    print(f"  {query}")

    if llm_draft:
        print(f"\n{_YELLOW}LLM Draft (for reference):{_RESET}")
        for line in llm_draft.splitlines():
            print(f"  {line}")

    print(f"\n{_CYAN}{separator}{_RESET}")
    print(
        f"{_CYAN}Please review the query above and provide a response.{_RESET}\n"
        f"{_CYAN}The user is waiting.{_RESET}"
    )
    print(f"{_CYAN}{separator}{_RESET}")
