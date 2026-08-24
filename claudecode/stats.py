"""Usage accounting for a review run.

Every provider accumulates into one of these, so the summary table reads the same
whichever backend produced the findings. Providers that cannot report a figure leave
it unset rather than reporting zero -- "no cost reported" and "cost was zero" are
different claims, and a table that renders $0.00 for a local GPU is a lie.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class UsageStats:
    """Token and cost totals, accumulated across every LLM call in a run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: Optional[float] = None

    def record(self, input_tokens: Any = 0, output_tokens: Any = 0,
               cached_input_tokens: Any = 0, cost_usd: Optional[float] = None) -> None:
        """Add one call's usage. Missing or unparseable figures count as zero."""
        self.calls += 1
        self.input_tokens += _as_int(input_tokens)
        self.output_tokens += _as_int(output_tokens)
        self.cached_input_tokens += _as_int(cached_input_tokens)
        if cost_usd is not None:
            try:
                self.cost_usd = (self.cost_usd or 0.0) + float(cost_usd)
            except (TypeError, ValueError):
                pass

    def merge(self, other: 'UsageStats') -> 'UsageStats':
        """Combine two accumulators, e.g. the audit call and the filtering calls."""
        combined = UsageStats(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )
        if self.cost_usd is not None or other.cost_usd is not None:
            combined.cost_usd = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return combined

    def as_dict(self) -> Dict[str, Any]:
        return {
            'llm_calls': self.calls,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cached_input_tokens': self.cached_input_tokens,
            'total_tokens': self.input_tokens + self.output_tokens,
            'cost_usd': self.cost_usd,
        }


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
