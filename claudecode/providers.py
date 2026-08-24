"""LLM provider abstraction for the security review.

The review makes exactly two LLM calls:

  1. the security audit itself  -- `run_security_audit()`
  2. per-finding false-positive filtering -- via the client from `create_filter_client()`

A provider owns both, so selecting a provider selects *every* model call. Everything
else in this package (hard exclusion rules, directory filtering, output assembly, PR
commenting) is deterministic and provider-agnostic.

Providers are imported lazily inside `create_provider()` so that a `spark` run never
touches the Anthropic SDK and an `anthropic` run never needs vLLM configuration.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - typing shim for Python < 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

    def runtime_checkable(cls):  # type: ignore
        return cls

from claudecode import prompts

SUPPORTED_PROVIDERS = ('anthropic', 'spark')
DEFAULT_PROVIDER = 'anthropic'


@runtime_checkable
class SecurityReviewProvider(Protocol):
    """What the security review needs from an LLM backend."""

    name: str
    display_name: str

    def validate(self) -> Tuple[bool, str]:
        """Check the provider is usable. Returns (ok, error_message)."""

    def build_prompt(self, pr_data: Dict[str, Any], pr_diff: Optional[str],
                     include_diff: bool = True,
                     custom_scan_instructions: Optional[str] = None) -> str:
        """Build the security audit prompt for this provider."""

    def run_security_audit(self, repo_dir: Path, prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Run the audit. Returns (success, error_message, results).

        On an over-long prompt the error message must be the sentinel
        ``"PROMPT_TOO_LONG"`` so the caller can retry with a smaller prompt.
        """

    def create_filter_client(self) -> Optional[Any]:
        """Client for false-positive filtering, or None if unavailable.

        Must expose `validate_api_access()` and `analyze_single_finding()`.
        """


class AnthropicProvider:
    """Claude Code (agentic, with repository tools) + the Anthropic Messages API.

    This is a thin adapter over the pre-existing implementation; behaviour is unchanged.
    """

    name = 'anthropic'
    display_name = 'Claude Code'

    def __init__(self, timeout_minutes: Optional[int] = None):
        # Imported here (not at module import time) so `mock.patch` targets in
        # github_action_audit keep working and to avoid a circular import.
        from claudecode import github_action_audit

        self.runner = github_action_audit.SimpleClaudeRunner(timeout_minutes)

    def validate(self) -> Tuple[bool, str]:
        return self.runner.validate_claude_available()

    def build_prompt(self, pr_data, pr_diff, include_diff=True, custom_scan_instructions=None) -> str:
        return prompts.get_security_audit_prompt(
            pr_data, pr_diff,
            include_diff=include_diff,
            custom_scan_instructions=custom_scan_instructions,
        )

    def run_security_audit(self, repo_dir: Path, prompt: str):
        return self.runner.run_security_audit(repo_dir, prompt)

    def create_filter_client(self) -> Optional[Any]:
        import os

        from claudecode.claude_api_client import ClaudeAPIClient

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return None
        return ClaudeAPIClient(api_key=api_key)


def create_provider(provider_name: Optional[str] = None,
                    timeout_minutes: Optional[int] = None) -> SecurityReviewProvider:
    """Build the provider named by `provider_name` (defaults to `anthropic`).

    Raises:
        ValueError: if the name is not one of SUPPORTED_PROVIDERS.
    """
    name = (provider_name or DEFAULT_PROVIDER).strip().lower()

    if name == 'anthropic':
        return AnthropicProvider(timeout_minutes=timeout_minutes)
    if name == 'spark':
        from claudecode.spark import SparkProvider

        return SparkProvider()

    raise ValueError(
        f"Unsupported provider: {provider_name!r}. "
        f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
    )
