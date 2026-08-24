# Claude Code Security Reviewer

An AI-powered security review GitHub Action that analyzes code changes for security vulnerabilities. See the original blog post [here](https://www.anthropic.com/news/automate-security-reviews-with-claude-code) for more details.

This is a fork of [`anthropics/claude-code-security-review`](https://github.com/anthropics/claude-code-security-review) that adds **pluggable LLM providers**. Pick one with the `provider` input:

| Provider | Backend | Requires |
|---|---|---|
| `anthropic` (default) | Claude Code, agentic with repository tools | `ANTHROPIC_API_KEY` |
| `spark` | Qwen3 on a DGX Spark, via an OpenAI-compatible vLLM endpoint | `VLLM_BASE_URL`, `VLLM_MODEL` |

`provider` defaults to `anthropic`, so existing workflows keep working unchanged.

## Features

- **AI-Powered Analysis**: Uses Claude's advanced reasoning to detect security vulnerabilities with deep semantic understanding
- **Diff-Aware Scanning**: For PRs, only analyzes changed files
- **PR Comments**: Automatically comments on PRs with security findings
- **Contextual Understanding**: Goes beyond pattern matching to understand code semantics
- **Language Agnostic**: Works with any programming language
- **False Positive Filtering**: Advanced filtering to reduce noise and focus on real vulnerabilities
- **Pluggable Providers**: Run against Anthropic's Claude Code or a self-hosted Qwen3 on vLLM

## Quick Start

Add this to your repository's `.github/workflows/security.yml`:

```yaml
name: Security Review

permissions:
  pull-requests: write  # Needed for leaving PR comments
  contents: read

on:
  pull_request:

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          fetch-depth: 2
      
      - uses: mohemian/claude-code-security-review@main
        with:
          comment-pr: true
          claude-api-key: ${{ secrets.CLAUDE_API_KEY }}
```

## Providers

### Anthropic (default)

Uses the existing Claude Code implementation: Claude runs as an agent with repository
exploration tools, so it can read files beyond the diff. This is the default and needs no
new configuration.

```yaml
- uses: mohemian/claude-code-security-review@main
  with:
    provider: anthropic
    claude-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

`ANTHROPIC_API_KEY` may also be supplied through the environment:

```yaml
- uses: mohemian/claude-code-security-review@main
  with:
    provider: anthropic
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

### DGX Spark / Qwen3

Talks directly to an OpenAI-compatible vLLM endpoint (`POST $VLLM_BASE_URL/chat/completions`).
It does **not** require Claude Code, the Anthropic SDK, or an Anthropic API key — neither is
installed or read when `provider: spark`.

```yaml
jobs:
  security:
    runs-on: [self-hosted, dgx-spark]
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          fetch-depth: 2

      - uses: mohemian/claude-code-security-review@main
        with:
          provider: spark
        env:
          VLLM_BASE_URL: http://localhost:8000/v1
          VLLM_MODEL: Qwen3-...
```

The same settings are also available as action inputs (`vllm-base-url`, `vllm-model`,
`vllm-api-key`), which take precedence over the environment variables.

Unlike the Anthropic provider, Spark has **no tool access**: the model receives the PR
metadata and diff in a single prompt and must answer with the same finding JSON. Its output
is treated as untrusted and validated against the finding schema before anything is posted
to GitHub; malformed output fails the run rather than being silently dropped.

Both providers cover *all* LLM calls, including per-finding false-positive filtering — the
Spark path never falls back to Anthropic.

#### Anthropic credentials

\* One of `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` is required. They are not
interchangeable, because the two Anthropic call sites authenticate differently:

| | Claude Code CLI (the audit) | Messages API (false-positive filtering) |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | yes |
| `CLAUDE_CODE_OAUTH_TOKEN` | yes | **no** — the API rejects it with `401` |

With an OAuth token the audit runs normally and false-positive filtering falls back to the
deterministic hard exclusion rules. The action emits a warning when this happens, so a
noisier-than-usual report has a visible cause rather than looking like a clean filter pass.

#### Reasoning models and the output budget

vLLM returns a reasoning model's trace in a separate `reasoning` field, but that trace is
generated *first* and counts against `max_tokens`. If the budget runs out mid-trace, the
response comes back with `finish_reason: "length"` and **no content at all** — the model
never reached its answer.

Measured on Qwen3.8-27B reviewing a ~40k-token diff: with reasoning on it consumed a full
16k-token budget and produced nothing, taking over ten minutes. With
`VLLM_ENABLE_THINKING=false` the same review completed in well under a minute.

The default budget is therefore `32768`, and a truncated response is always reported as an
error rather than parsed as a partial report. If you see one, either raise
`VLLM_MAX_TOKENS` or set `VLLM_ENABLE_THINKING=false`.

> Whether reasoning improves finding quality enough to pay for is a benchmarking question,
> not a settled one. The knob is there so you can measure it.

### Provider environment variables

| Variable | Provider | Required | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | `anthropic` | Yes* | Anthropic API key, enabled for both the Claude API and Claude Code. Equivalent to the `claude-api-key` input. |
| `CLAUDE_CODE_OAUTH_TOKEN` | `anthropic` | Yes* | Claude Code subscription OAuth token (`sk-ant-oat01-...`), as an alternative to the API key. Equivalent to the `claude-code-oauth-token` input. **Runs the audit but not the filtering** — see below. |
| `CLAUDE_MODEL` | `anthropic` | No | Claude model override. Equivalent to the `claude-model` input. |
| `VLLM_BASE_URL` | `spark` | Yes | OpenAI-compatible base URL, e.g. `http://localhost:8000/v1`. |
| `VLLM_MODEL` | `spark` | Yes | Model name as served by vLLM, e.g. `Qwen3-32B`. Passed through verbatim; no model name is hard-coded. |
| `VLLM_API_KEY` | `spark` | No | Bearer token, if the endpoint requires one. A local DGX Spark deployment usually does not. |
| `VLLM_TIMEOUT` | `spark` | No | Per-request timeout in seconds (default `900`). |
| `VLLM_MAX_TOKENS` | `spark` | No | Output budget per call (default `32768`). A reasoning model spends this on its reasoning trace *before* writing any answer, so it needs headroom — see below. |
| `VLLM_ENABLE_THINKING` | `spark` | No | Set `false` to switch the model's reasoning off (`chat_template_kwargs.enable_thinking`). Unset leaves the served model's own default alone. |

Configuration is validated before any model call, and errors name exactly what is missing.
An unsupported `provider` value fails immediately with the list of supported values.

## Security Considerations

This action is not hardened against prompt injection attacks and should only be used to review trusted PRs. We recommend [configuring your repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#controlling-changes-from-forks-to-workflows-in-public-repositories) to use the "Require approval for all external contributors" option to ensure workflows only run after a maintainer has reviewed the PR.

## Configuration Options

### Action Inputs

| Input | Description | Default | Required |
|-------|-------------|---------|----------|
| `provider` | LLM provider to use: `anthropic` or `spark` | `anthropic` | No |
| `claude-api-key` | Anthropic Claude API key for security analysis. <br>*Note*: This API key needs to be enabled for both the Claude API and Claude Code usage. | None | Yes for `provider: anthropic` |
| `comment-pr` | Whether to comment on PRs with findings | `true` | No |
| `upload-results` | Whether to upload results as artifacts | `true` | No |
| `exclude-directories` | Comma-separated list of directories to exclude from scanning | None | No |
| `claude-model` | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1. | `claude-opus-4-1-20250805` | No |
| `claudecode-timeout` | Timeout for ClaudeCode analysis in minutes | `20` | No |
| `run-every-commit` | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits. | `false` | No |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file | None | No |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt | None | No |
| `vllm-base-url` | vLLM base URL; overrides `VLLM_BASE_URL` | None | Yes for `provider: spark` (or set `VLLM_BASE_URL`) |
| `vllm-model` | Model served by vLLM; overrides `VLLM_MODEL` | None | Yes for `provider: spark` (or set `VLLM_MODEL`) |
| `vllm-api-key` | Bearer token for the vLLM endpoint; overrides `VLLM_API_KEY` | None | No |

### Run summary

At the end of every run the action writes a table to the GitHub Actions **job summary**
(the run's landing page — no need to open the logs):

| Metric | Value |
|---|---|
| Provider | `spark` |
| Model | `unsloth/Qwen3.8-27B-NVFP4` |
| Duration (total) | 1m 14s |
| — security audit | 46s |
| — false-positive filtering | 26s |
| LLM calls | 4 |
| Input tokens | 7,441 |
| Output tokens | 991 |
| Cached input tokens | 0 |
| Total tokens | 8,432 |
| Cost | not reported |
| Findings reported by model | 3 |
| Findings after filtering | **3** |
| Findings excluded | 0 |

The same figures are in the results JSON under `run_stats`, so they can be consumed by
later workflow steps.

Cost shows **not reported** rather than `$0.00` for providers that do not price their
calls — a self-hosted vLLM has no per-call cost, and printing zero would be a claim rather
than an absence. Only Claude Code reports a real figure (`total_cost_usd`).

If the scan fails, the summary shows the error instead of the table.

### Action Outputs

| Output | Description |
|--------|-------------|
| `findings-count` | Total number of security findings |
| `results-file` | Path to the results JSON file |

## How It Works

### Architecture

```
claudecode/
├── github_action_audit.py     # Main audit script for GitHub Actions
├── providers.py               # Provider protocol, factory, Anthropic provider
├── spark.py                   # Spark provider: vLLM client + finding validation
├── prompts.py                 # Security audit prompt templates
├── filter_prompts.py          # Shared false positive filtering prompts
├── findings_filter.py         # False positive filtering logic
├── claude_api_client.py       # Anthropic API client for false positive filtering
├── json_parser.py             # Robust JSON parsing utilities
├── requirements.txt           # Python dependencies
├── requirements-anthropic.txt # Extra dependencies for provider: anthropic
├── test_*.py                  # Test suites
└── evals/                     # Eval tooling to test CC on arbitrary PRs
```

Every LLM call goes through a provider:

```
        Security review
              |
              v
        create_provider()
         /            \
AnthropicProvider   SparkProvider
        |                 |
   Claude Code           vLLM
        |                 |
     Claude              Qwen3
```

A provider owns both model calls — the security audit and false-positive filtering — so
selecting a provider selects every LLM call. Everything else (hard exclusion rules,
directory filtering, output assembly, PR commenting) is deterministic and provider-agnostic.

### Workflow

1. **PR Analysis**: When a pull request is opened, Claude analyzes the diff to understand what changed
2. **Contextual Review**: Claude examines the code changes in context, understanding the purpose and potential security implications
3. **Finding Generation**: Security issues are identified with detailed explanations, severity ratings, and remediation guidance
4. **False Positive Filtering**: Advanced filtering removes low-impact or false positive prone findings to reduce noise
5. **PR Comments**: Findings are posted as review comments on the specific lines of code

## Security Analysis Capabilities

### Types of Vulnerabilities Detected

- **Injection Attacks**: SQL injection, command injection, LDAP injection, XPath injection, NoSQL injection, XXE
- **Authentication & Authorization**: Broken authentication, privilege escalation, insecure direct object references, bypass logic, session flaws
- **Data Exposure**: Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations
- **Cryptographic Issues**: Weak algorithms, improper key management, insecure random number generation
- **Input Validation**: Missing validation, improper sanitization, buffer overflows
- **Business Logic Flaws**: Race conditions, time-of-check-time-of-use (TOCTOU) issues
- **Configuration Security**: Insecure defaults, missing security headers, permissive CORS
- **Supply Chain**: Vulnerable dependencies, typosquatting risks
- **Code Execution**: RCE via deserialization, pickle injection, eval injection
- **Cross-Site Scripting (XSS)**: Reflected, stored, and DOM-based XSS

### False Positive Filtering

The tool automatically excludes a variety of low-impact and false positive prone findings to focus on high-impact vulnerabilities:
- Denial of Service vulnerabilities
- Rate limiting concerns
- Memory/CPU exhaustion issues
- Generic input validation without proven impact
- Open redirect vulnerabilities

The false positive filtering can also be tuned as needed for a given project's security goals.

### Benefits Over Traditional SAST

- **Contextual Understanding**: Understands code semantics and intent, not just patterns
- **Lower False Positives**: AI-powered analysis reduces noise by understanding when code is actually vulnerable
- **Detailed Explanations**: Provides clear explanations of why something is a vulnerability and how to fix it
- **Adaptive Learning**: Can be customized with organization-specific security requirements

## Installation & Setup

### GitHub Actions

Follow the Quick Start guide above. The action handles all dependencies automatically.

### Local Development

To run the security scanner locally against a specific PR, see the [evaluation framework documentation](claudecode/evals/README.md).

<a id="security-review-slash-command"></a>

## Claude Code Integration: /security-review Command 

By default, Claude Code ships a `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) that provides the same security analysis capabilities as the GitHub Action workflow, but integrated directly into your Claude Code development environment. To use this, simply run `/security-review` to perform a comprehensive security review of all pending changes.

### Customizing the Command

The default `/security-review` command is designed to work well in most cases, but it can also be customized based on your specific security needs. To do so: 

1. Copy the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file from this repository to your project's `.claude/commands/` folder. 
2. Edit `security-review.md` to customize the security analysis. For example, you could add additional organization-specific directions to the false positive filtering instructions. 

## Custom Scanning Configuration

It is also possible to configure custom scanning and false positive filtering instructions, see the [`docs/`](docs/) folder for more details.  

## Testing

### Unit tests

```bash
pip install pytest && pip install -r claudecode/requirements.txt
pip install -r claudecode/requirements-anthropic.txt   # only for the anthropic tests
PYTHONPATH=$PWD pytest claudecode -q
```

### Dry-running a provider against a real PR

`github_action_audit.py` is a plain script: give it a PR and it prints the findings JSON to
stdout. Nothing is posted to GitHub (PR comments are a separate action step), so this is
safe to point at any PR you can read.

```bash
GITHUB_TOKEN=$(gh auth token) \
GITHUB_REPOSITORY=owner/repo PR_NUMBER=123 \
LLM_PROVIDER=spark ENABLE_CLAUDE_FILTERING=true \
VLLM_BASE_URL=http://localhost:8000/v1 VLLM_MODEL=Qwen3-32B \
REPO_PATH=$PWD PYTHONPATH=$PWD python claudecode/github_action_audit.py
```

Swap in `LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-...` to dry-run the other provider
(that one needs the `claude` CLI on PATH and a checkout of the repo under review at
`REPO_PATH`, since Claude Code explores it).

### Testing Spark without a DGX Spark

`scripts/fake-vllm-server.py` stands in for vLLM and speaks enough of the OpenAI API to
exercise the real HTTP path, including the failure modes:

```bash
python3 scripts/fake-vllm-server.py --port 8000 --mode findings
# modes: findings | empty | malformed | prose | toolong | flaky
```

Then run the dry-run command above against `http://localhost:8000/v1`. Expected results:

| Mode | Expected |
|---|---|
| `findings` | exit 1, one HIGH finding, two vLLM calls (audit + filter) |
| `empty` | exit 0, no findings |
| `malformed` | exit 1, `Invalid security report from model: findings[0] has severity ...` |
| `prose` | exit 1, `Could not parse JSON from vLLM response` |
| `toolong` | two audit calls, then an error telling you to exclude directories |
| `flaky` | HTTP 500 retried, then a normal result |

### Positive control

A scanner that finds nothing looks identical to a scanner that is broken. `examples/vulnerable_sample.py`
(on the `test/planted-vuln` branch, PR #2) carries a SQL injection, a command injection and
a path traversal for exactly that reason — point a provider at that PR and confirm all three
come back before trusting a clean result on real code.

Note that the false-positive filter reads each finding's file from disk, so `REPO_PATH` must
point at a checkout of the **PR head**. If the file is missing the filter is told so, and
tends to discard the finding — a clean report from the wrong checkout means nothing.

Both providers have been run against that fixture and found all three issues. One known
difference: on this sample Qwen3 reported line numbers roughly two lines earlier than the
vulnerable expression (pointing at comments and setup), where Claude Code was exact. Line
numbers drive review-comment placement, and GitHub rejects a comment on a line outside the
diff, so this is worth measuring before relying on Spark for inline comments.


Run the test suite to validate functionality:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

For issues or questions:
- Open an issue in this repository
- Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information

## License

MIT License - see [LICENSE](LICENSE) file for details.
