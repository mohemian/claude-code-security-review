"""Spark provider: security review against a local Qwen3 served by vLLM.

vLLM exposes an OpenAI-compatible API, so this is a plain `POST {base}/chat/completions`
with the `requests` dependency the action already ships. There is no agent loop and no
tool access: the model gets the PR metadata + diff in one prompt and must answer with the
same finding JSON the Anthropic path produces.

Model output is untrusted. `sanitize_audit_results()` enforces the schema, whitelists
fields, clamps lengths and rejects unsafe file paths before anything reaches GitHub.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from claudecode import filter_prompts
from claudecode.json_parser import parse_json_with_fallbacks
from claudecode.logger import get_logger
from claudecode.stats import UsageStats
from claudecode import prompts

logger = get_logger(__name__)

DEFAULT_REQUEST_TIMEOUT = 900  # seconds; a full-diff audit on a local GPU is slow
DEFAULT_MAX_RETRIES = 3

# Output budget per call. A reasoning model spends this on its reasoning trace *before*
# it writes any answer, so it has to be generous: Qwen3 will happily burn 16k tokens
# thinking about a large diff and never reach the JSON.
DEFAULT_MAX_TOKENS = 32768
PROMPT_TOO_LONG = 'PROMPT_TOO_LONG'

# Retry-on-oversize fallback: how much diff to keep when the model refused the full one.
# ponytail: flat character budget, not tokens. Swap in a tokenizer if it proves too coarse.
TRUNCATED_DIFF_CHARS = 60_000

# Caps on model-authored strings that end up in PR comments.
MAX_FIELD_CHARS = 4000
MAX_PATH_CHARS = 500
MAX_FINDINGS = 100

REQUIRED_FINDING_FIELDS = ('file', 'line', 'severity', 'description')
VALID_SEVERITIES = ('HIGH', 'MEDIUM', 'LOW')

_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_CONTEXT_LENGTH_HINTS = (
    'maximum context length', 'context length', 'longer than the maximum',
    'reduce the length', 'too long',
)


class VLLMResponseError(ValueError):
    """Raised when the model returns something that is not a usable security report."""


def strip_reasoning(text: str) -> str:
    """Drop Qwen3 `<think>` blocks so they cannot confuse the JSON extractor."""
    return _THINK_BLOCK.sub('', text).strip()


class VLLMClient:
    """Minimal OpenAI-compatible chat client for a vLLM server."""

    def __init__(self,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout_seconds: Optional[int] = None,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 max_tokens: Optional[int] = None,
                 enable_thinking: Optional[bool] = None):
        self.base_url = (base_url or '').strip().rstrip('/')
        self.model = (model or '').strip()
        self.api_key = api_key or None
        self.timeout_seconds = timeout_seconds or DEFAULT_REQUEST_TIMEOUT
        self.max_retries = max_retries
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        # Shared by the audit and the filtering calls, which use the same client.
        self.usage = UsageStats()
        # None leaves the served model's own default alone.
        self.enable_thinking = enable_thinking
        # Set to False after the server rejects `response_format`.
        self.json_mode = True

    @classmethod
    def from_env(cls) -> 'VLLMClient':
        """Build a client from VLLM_BASE_URL / VLLM_MODEL / VLLM_API_KEY."""
        timeout = os.environ.get('VLLM_TIMEOUT')
        max_tokens = os.environ.get('VLLM_MAX_TOKENS')
        thinking = os.environ.get('VLLM_ENABLE_THINKING')
        return cls(
            base_url=os.environ.get('VLLM_BASE_URL'),
            model=os.environ.get('VLLM_MODEL'),
            api_key=os.environ.get('VLLM_API_KEY'),
            timeout_seconds=int(timeout) if timeout and timeout.isdigit() else None,
            max_tokens=int(max_tokens) if max_tokens and max_tokens.isdigit() else None,
            enable_thinking=(thinking.strip().lower() in ('1', 'true', 'yes')
                             if thinking else None),
        )

    @property
    def endpoint(self) -> str:
        return f'{self.base_url}/chat/completions'

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration only -- no network call, no secrets logged."""
        missing = []
        if not self.base_url:
            missing.append('VLLM_BASE_URL (e.g. http://localhost:8000/v1)')
        if not self.model:
            missing.append('VLLM_MODEL (the model name served by vLLM)')
        if missing:
            return False, (
                'Spark provider is not configured. Missing: ' + ', '.join(missing)
            )
        if not self.base_url.startswith(('http://', 'https://')):
            return False, f'VLLM_BASE_URL must start with http:// or https://, got: {self.base_url}'
        return True, ''

    def validate_api_access(self) -> Tuple[bool, str]:
        """FindingsFilter calls this before enabling LLM filtering."""
        return self.validate()

    def chat(self, messages: List[Dict[str, str]],
             max_tokens: Optional[int] = None,
             temperature: float = 0.0) -> Tuple[bool, str, str]:
        """Call the chat completions endpoint.

        Returns:
            (success, content, error_message). `error_message` is the sentinel
            PROMPT_TOO_LONG when the server rejected the prompt as oversized.
        """
        ok, error = self.validate()
        if not ok:
            return False, '', error

        max_tokens = max_tokens or self.max_tokens
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        last_error = ''
        for attempt in range(self.max_retries):
            payload: Dict[str, Any] = {
                'model': self.model,
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'stream': False,
            }
            if self.json_mode:
                payload['response_format'] = {'type': 'json_object'}
            if self.enable_thinking is not None:
                payload['chat_template_kwargs'] = {'enable_thinking': self.enable_thinking}

            try:
                response = requests.post(
                    self.endpoint, headers=headers, json=payload,
                    timeout=self.timeout_seconds,
                )
            except requests.exceptions.Timeout:
                last_error = f'vLLM request timed out after {self.timeout_seconds}s'
                logger.warning(f'{last_error} (attempt {attempt + 1}/{self.max_retries})')
                continue
            except requests.exceptions.RequestException as e:
                last_error = f'vLLM request failed: {e}'
                logger.warning(f'{last_error} (attempt {attempt + 1}/{self.max_retries})')
                time.sleep(2 * attempt)
                continue

            if response.status_code >= 500:
                last_error = f'vLLM returned HTTP {response.status_code}: {response.text[:300]}'
                logger.warning(f'{last_error} (attempt {attempt + 1}/{self.max_retries})')
                time.sleep(2 * attempt)
                continue

            if response.status_code >= 400:
                body = response.text[:1000]
                lowered = body.lower()
                if self.json_mode and ('response_format' in lowered or 'guided' in lowered):
                    logger.warning('vLLM rejected response_format; retrying without JSON mode')
                    self.json_mode = False
                    continue
                if any(hint in lowered for hint in _CONTEXT_LENGTH_HINTS):
                    return False, '', PROMPT_TOO_LONG
                return False, '', f'vLLM returned HTTP {response.status_code}: {body}'

            try:
                data = response.json()
            except ValueError:
                return False, '', f'vLLM returned non-JSON body: {response.text[:300]}'

            usage = data.get('usage') if isinstance(data, dict) else None
            if isinstance(usage, dict):
                # vLLM is local, so there is no cost to report -- leave it unset.
                self.usage.record(
                    input_tokens=usage.get('prompt_tokens'),
                    output_tokens=usage.get('completion_tokens'),
                )

            success, content, error = _extract_content(data, max_tokens)
            if success:
                return True, content, ''
            return False, '', error

        return False, '', f'vLLM call failed after {self.max_retries} attempts: {last_error}'


def _extract_content(data: Any, max_tokens: int) -> Tuple[bool, str, str]:
    """Pull the assistant message out of an OpenAI-compatible response."""
    if not isinstance(data, dict):
        return False, '', 'vLLM response was not a JSON object'
    if 'error' in data and data.get('error'):
        return False, '', f'vLLM error: {str(data["error"])[:300]}'

    choices = data.get('choices')
    if not isinstance(choices, list) or not choices:
        return False, '', 'vLLM response contained no choices'

    message = choices[0].get('message') if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return False, '', 'vLLM response choice contained no message'

    content = message.get('content')

    # A reasoning model emits its trace first, in a separate field. If the budget runs
    # out mid-trace there is no answer at all, and the raw symptom ("no content") says
    # nothing useful -- so name the actual cause and the two ways out.
    if choices[0].get('finish_reason') == 'length':
        reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
        if not (content or '').strip() and reasoning:
            return False, '', (
                f'the model spent its entire {max_tokens}-token output budget on its '
                f'reasoning trace without producing an answer. Raise VLLM_MAX_TOKENS, or '
                f'set VLLM_ENABLE_THINKING=false to skip reasoning entirely.'
            )
        return False, '', (
            f'the model hit the {max_tokens}-token output budget, so its report is '
            f'truncated and cannot be trusted. Raise VLLM_MAX_TOKENS.'
        )

    if not isinstance(content, str) or not content.strip():
        return False, '', 'vLLM response message contained no content'

    return True, strip_reasoning(content), ''


def _clean_text(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    """Coerce model output to a bounded, control-character-free string."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = ''.join(ch for ch in text if ch == '\n' or ch == '\t' or ord(ch) >= 32)
    return text[:limit]


def _sanitize_finding(finding: Any, index: int) -> Dict[str, Any]:
    """Validate and normalise one model-produced finding.

    Raises:
        VLLMResponseError: if the finding is unusable.
    """
    if not isinstance(finding, dict):
        raise VLLMResponseError(f'findings[{index}] is not an object')

    for field in REQUIRED_FINDING_FIELDS:
        if finding.get(field) in (None, ''):
            raise VLLMResponseError(f'findings[{index}] is missing required field {field!r}')

    file_path = _clean_text(finding['file'], MAX_PATH_CHARS).strip().lstrip('/')
    if not file_path or '..' in Path(file_path).parts:
        raise VLLMResponseError(f'findings[{index}] has an unsafe file path: {finding["file"]!r}')

    try:
        line = int(finding['line'])
    except (TypeError, ValueError):
        raise VLLMResponseError(f'findings[{index}] has a non-numeric line: {finding["line"]!r}')
    if line < 1:
        line = 1

    severity = _clean_text(finding['severity'], 20).strip().upper()
    if severity not in VALID_SEVERITIES:
        raise VLLMResponseError(
            f'findings[{index}] has severity {finding["severity"]!r}; '
            f'expected one of {", ".join(VALID_SEVERITIES)}'
        )

    clean: Dict[str, Any] = {
        'file': file_path,
        'line': line,
        'severity': severity,
        'description': _clean_text(finding['description']),
    }
    for field in ('category', 'exploit_scenario', 'recommendation'):
        if finding.get(field):
            clean[field] = _clean_text(finding[field])

    try:
        clean['confidence'] = max(0.0, min(1.0, float(finding.get('confidence', 0.8))))
    except (TypeError, ValueError):
        clean['confidence'] = 0.8

    return clean


def sanitize_audit_results(parsed: Any) -> Dict[str, Any]:
    """Validate a parsed model response against the finding schema.

    Raises:
        VLLMResponseError: if the response is not a usable security report.
    """
    if not isinstance(parsed, dict):
        raise VLLMResponseError('model response was not a JSON object')

    findings = parsed.get('findings')
    if findings is None:
        raise VLLMResponseError("model response is missing the 'findings' key")
    if not isinstance(findings, list):
        raise VLLMResponseError("model response 'findings' must be a list")
    if len(findings) > MAX_FINDINGS:
        raise VLLMResponseError(
            f'model returned {len(findings)} findings, above the {MAX_FINDINGS} limit'
        )

    clean_findings = [_sanitize_finding(f, i) for i, f in enumerate(findings)]

    summary = parsed.get('analysis_summary')
    if not isinstance(summary, dict):
        summary = {}

    return {
        'findings': clean_findings,
        'analysis_summary': {
            'files_reviewed': summary.get('files_reviewed', 0),
            'high_severity': len([f for f in clean_findings if f['severity'] == 'HIGH']),
            'medium_severity': len([f for f in clean_findings if f['severity'] == 'MEDIUM']),
            'low_severity': len([f for f in clean_findings if f['severity'] == 'LOW']),
            'review_completed': True,
        },
    }


class SparkFilterClient:
    """False-positive filtering through vLLM.

    Same prompts as the Anthropic path (see `filter_prompts`), so filtering behaviour
    does not silently diverge between providers.
    """

    def __init__(self, client: VLLMClient, custom_filtering_instructions: Optional[str] = None):
        self.client = client
        self.custom_filtering_instructions = custom_filtering_instructions

    def validate_api_access(self) -> Tuple[bool, str]:
        return self.client.validate_api_access()

    def analyze_single_finding(self, finding: Dict[str, Any],
                               pr_context: Optional[Dict[str, Any]] = None,
                               custom_filtering_instructions: Optional[str] = None
                               ) -> Tuple[bool, Dict[str, Any], str]:
        instructions = custom_filtering_instructions or self.custom_filtering_instructions
        messages = [
            {'role': 'system', 'content': filter_prompts.generate_system_prompt()},
            {'role': 'user', 'content': filter_prompts.generate_single_finding_prompt(
                finding, pr_context, instructions)},
        ]

        success, content, error = self.client.chat(messages)
        if not success:
            return False, {}, error

        parsed_ok, parsed = parse_json_with_fallbacks(content, 'vLLM filter response')
        if not parsed_ok or not isinstance(parsed, dict):
            return False, {}, 'Failed to parse JSON from vLLM filter response'

        # Unknown/garbled verdicts must not silently drop a finding.
        if not isinstance(parsed.get('keep_finding'), bool):
            return False, {}, "vLLM filter response is missing a boolean 'keep_finding'"

        return True, parsed, ''


class SparkProvider:
    """Qwen3 on a DGX Spark, served by vLLM. No Claude Code, no Anthropic credentials."""

    name = 'spark'
    display_name = 'Spark provider'

    def __init__(self, client: Optional[VLLMClient] = None):
        self.client = client or VLLMClient.from_env()

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def usage(self) -> UsageStats:
        return self.client.usage

    def validate(self) -> Tuple[bool, str]:
        return self.client.validate()

    def build_prompt(self, pr_data, pr_diff, include_diff=True, custom_scan_instructions=None) -> str:
        """Build a tool-free audit prompt.

        `include_diff=False` means the caller is retrying after PROMPT_TOO_LONG. Spark has
        no way to fetch the code itself, so instead of dropping the diff (which would yield
        a confident, empty review of nothing) it reviews a truncated diff.
        """
        if not include_diff and pr_diff:
            pr_diff = (
                pr_diff[:TRUNCATED_DIFF_CHARS]
                + '\n\n[diff truncated: it exceeded the model context window. '
                  'Only the portion above is available for review.]'
            )
            include_diff = True
        return prompts.get_security_audit_prompt(
            pr_data, pr_diff,
            include_diff=include_diff,
            custom_scan_instructions=custom_scan_instructions,
            agentic=False,
        )

    def run_security_audit(self, repo_dir: Path, prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Run the audit as a single completion. `repo_dir` is unused (no tool access)."""
        messages = [
            {'role': 'system', 'content':
                'You are a senior security engineer. Respond with a single JSON object '
                'matching the requested schema and nothing else.'},
            {'role': 'user', 'content': prompt},
        ]

        success, content, error = self.client.chat(messages)
        if not success:
            return False, error, {}

        parsed_ok, parsed = parse_json_with_fallbacks(content, 'vLLM security audit response')
        if not parsed_ok:
            return False, f'Could not parse JSON from vLLM response: {str(parsed)[:500]}', {}

        try:
            return True, '', sanitize_audit_results(parsed)
        except VLLMResponseError as e:
            return False, f'Invalid security report from model: {e}', {}

    def create_filter_client(self) -> Optional[Any]:
        return SparkFilterClient(self.client)
