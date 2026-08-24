#!/usr/bin/env python3
"""Tests for the LLM provider abstraction: selection, configuration and isolation."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claudecode.github_action_audit import (
    ConfigurationError,
    get_provider_name,
    initialize_clients,
    initialize_findings_filter,
)
from claudecode.providers import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    AnthropicProvider,
    create_provider,
)
from claudecode.spark import SparkProvider

SPARK_ENV = {'VLLM_BASE_URL': 'http://localhost:8000/v1', 'VLLM_MODEL': 'Qwen3-32B'}


class TestProviderSelection:
    """provider=anthropic -> AnthropicProvider, provider=spark -> SparkProvider."""

    def test_default_is_anthropic(self):
        assert DEFAULT_PROVIDER == 'anthropic'
        assert isinstance(create_provider(), AnthropicProvider)
        assert isinstance(create_provider(None), AnthropicProvider)

    def test_anthropic_selected_explicitly(self):
        provider = create_provider('anthropic')
        assert isinstance(provider, AnthropicProvider)
        assert provider.name == 'anthropic'

    def test_spark_selected(self):
        with patch.dict(os.environ, SPARK_ENV):
            provider = create_provider('spark')
        assert isinstance(provider, SparkProvider)
        assert provider.name == 'spark'

    def test_provider_name_is_case_and_whitespace_insensitive(self):
        with patch.dict(os.environ, SPARK_ENV):
            assert isinstance(create_provider('  SPARK '), SparkProvider)

    def test_invalid_provider_raises_clear_error(self):
        with pytest.raises(ValueError) as exc_info:
            create_provider('openai')
        message = str(exc_info.value)
        assert "Unsupported provider: 'openai'" in message
        assert 'anthropic' in message and 'spark' in message

    def test_get_provider_name_defaults_to_anthropic(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_provider_name() == 'anthropic'

    def test_get_provider_name_reads_environment(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'spark'}):
            assert get_provider_name() == 'spark'

    def test_get_provider_name_rejects_unknown(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'llama'}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_provider_name()
        assert 'Unsupported provider' in str(exc_info.value)
        assert 'anthropic, spark' in str(exc_info.value)

    @patch('claudecode.github_action_audit.GitHubActionClient')
    def test_initialize_clients_rejects_unknown_provider(self, mock_github):
        mock_github.return_value = Mock()
        with pytest.raises(ConfigurationError) as exc_info:
            initialize_clients('gemini')
        assert 'Unsupported provider' in str(exc_info.value)

    def test_every_supported_provider_is_constructible(self):
        with patch.dict(os.environ, SPARK_ENV):
            for name in SUPPORTED_PROVIDERS:
                assert create_provider(name).name == name


class TestSparkConfiguration:
    """provider=spark validates its own configuration, and only its own."""

    def test_valid_configuration(self):
        with patch.dict(os.environ, SPARK_ENV, clear=True):
            ok, error = create_provider('spark').validate()
        assert ok is True
        assert error == ''

    def test_missing_base_url(self):
        with patch.dict(os.environ, {'VLLM_MODEL': 'Qwen3-32B'}, clear=True):
            ok, error = create_provider('spark').validate()
        assert ok is False
        assert 'VLLM_BASE_URL' in error
        assert 'VLLM_MODEL' not in error

    def test_missing_model(self):
        with patch.dict(os.environ, {'VLLM_BASE_URL': 'http://localhost:8000/v1'}, clear=True):
            ok, error = create_provider('spark').validate()
        assert ok is False
        assert 'VLLM_MODEL' in error
        assert 'VLLM_BASE_URL' not in error

    def test_missing_everything_names_both(self):
        with patch.dict(os.environ, {}, clear=True):
            ok, error = create_provider('spark').validate()
        assert ok is False
        assert 'VLLM_BASE_URL' in error and 'VLLM_MODEL' in error

    def test_non_http_base_url_rejected(self):
        with patch.dict(os.environ, {'VLLM_BASE_URL': 'localhost:8000', 'VLLM_MODEL': 'q'}, clear=True):
            ok, error = create_provider('spark').validate()
        assert ok is False
        assert 'http://' in error


class TestAnthropicConfiguration:
    """provider=anthropic keeps validating Claude Code + ANTHROPIC_API_KEY as before."""

    @patch('claudecode.github_action_audit.subprocess.run')
    def test_missing_api_key_reported(self, mock_run):
        mock_run.return_value = Mock(returncode=0, stdout='1.0.0', stderr='')
        with patch.dict(os.environ, {}, clear=True):
            ok, error = create_provider('anthropic').validate()
        assert ok is False
        assert 'ANTHROPIC_API_KEY' in error

    @patch('claudecode.github_action_audit.subprocess.run')
    def test_valid_configuration(self, mock_run):
        mock_run.return_value = Mock(returncode=0, stdout='1.0.0', stderr='')
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-test'}, clear=True):
            ok, error = create_provider('anthropic').validate()
        assert ok is True

    @patch('claudecode.github_action_audit.subprocess.run')
    def test_does_not_require_vllm_configuration(self, mock_run):
        """provider=anthropic must not ask for VLLM_* at all."""
        mock_run.return_value = Mock(returncode=0, stdout='1.0.0', stderr='')
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-test'}, clear=True):
            provider = create_provider('anthropic')
            ok, error = provider.validate()
        assert ok is True
        assert 'VLLM' not in error


class TestProviderIsolation:
    """provider=spark must never need Anthropic credentials, Claude Code or the SDK."""

    def test_spark_validates_without_anthropic_key(self):
        with patch.dict(os.environ, SPARK_ENV, clear=True):
            assert 'ANTHROPIC_API_KEY' not in os.environ
            ok, _ = create_provider('spark').validate()
        assert ok is True

    def test_spark_filter_client_does_not_read_anthropic_key(self):
        with patch.dict(os.environ, SPARK_ENV, clear=True):
            client = create_provider('spark').create_filter_client()
        assert client is not None
        ok, _ = client.validate_api_access()
        assert ok is True

    def test_spark_audit_never_shells_out(self):
        """No `claude` subprocess on the Spark path."""
        with patch.dict(os.environ, SPARK_ENV, clear=True):
            provider = create_provider('spark')
        with patch('subprocess.run') as mock_run, \
                patch.object(provider.client, 'chat', return_value=(True, '{"findings": []}', '')):
            success, error, results = provider.run_security_audit(Path('.'), 'prompt')
        assert success is True
        mock_run.assert_not_called()

    def test_spark_filtering_is_enabled_without_anthropic_key(self):
        """The hidden-Anthropic-dependency regression: filtering must still run."""
        with patch.dict(os.environ, dict(SPARK_ENV, ENABLE_CLAUDE_FILTERING='true'), clear=True):
            provider = create_provider('spark')
            findings_filter = initialize_findings_filter(None, provider)
        assert findings_filter.use_claude_filtering is True
        assert findings_filter.claude_client is not None

    def test_anthropic_filtering_falls_back_without_key(self):
        with patch.dict(os.environ, {'ENABLE_CLAUDE_FILTERING': 'true'}, clear=True):
            provider = create_provider('anthropic')
            findings_filter = initialize_findings_filter(None, provider)
        assert findings_filter.use_claude_filtering is False

    def test_importing_the_audit_module_does_not_import_anthropic(self):
        """The Anthropic SDK must stay lazily imported so spark can run without it."""
        code = (
            'import claudecode.github_action_audit, claudecode.spark, claudecode.providers, sys; '
            "assert 'anthropic' not in sys.modules, 'anthropic was imported eagerly'"
        )
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, result.stderr

    def test_spark_module_does_not_reference_anthropic_credentials(self):
        source = (Path(__file__).resolve().parent / 'spark.py').read_text()
        assert 'ANTHROPIC_API_KEY' not in source
        assert 'import anthropic' not in source


class TestProviderPrompts:
    """Both providers reuse the shared prompt; only tool instructions differ."""

    pr_data = {
        'number': 7, 'title': 'Add login', 'user': 'alice', 'changed_files': 1,
        'additions': 10, 'deletions': 0, 'files': [{'filename': 'app.py'}],
        'head': {'repo': {'full_name': 'o/r'}},
    }

    def test_anthropic_prompt_keeps_tool_instructions(self):
        prompt = create_provider('anthropic').build_prompt(self.pr_data, 'diff-text')
        assert 'file search tools' in prompt
        assert 'diff-text' in prompt

    def test_spark_prompt_removes_tool_instructions(self):
        with patch.dict(os.environ, SPARK_ENV):
            prompt = create_provider('spark').build_prompt(self.pr_data, 'diff-text')
        assert 'file search tools' not in prompt
        assert 'no tools available' in prompt
        assert 'diff-text' in prompt

    def test_both_prompts_request_the_same_finding_schema(self):
        with patch.dict(os.environ, SPARK_ENV):
            spark_prompt = create_provider('spark').build_prompt(self.pr_data, 'd')
        anthropic_prompt = create_provider('anthropic').build_prompt(self.pr_data, 'd')
        for prompt in (spark_prompt, anthropic_prompt):
            assert '"findings"' in prompt
            assert '"exploit_scenario"' in prompt

    def test_spark_truncates_rather_than_dropping_the_diff(self):
        """include_diff=False is the too-long retry; Spark cannot fetch code itself."""
        big_diff = 'x' * 200_000
        with patch.dict(os.environ, SPARK_ENV):
            prompt = create_provider('spark').build_prompt(self.pr_data, big_diff, include_diff=False)
        assert 'diff truncated' in prompt
        assert len(prompt) < len(big_diff)
        assert 'x' * 1000 in prompt


class TestProviderProtocolSurface:
    """Every provider implements the whole surface the review depends on."""

    @pytest.mark.parametrize('name', SUPPORTED_PROVIDERS)
    def test_surface(self, name):
        with patch.dict(os.environ, SPARK_ENV):
            provider = create_provider(name)
        for attribute in ('name', 'display_name', 'validate', 'build_prompt',
                          'run_security_audit', 'create_filter_client'):
            assert hasattr(provider, attribute), f'{name} is missing {attribute}'


class TestPromptTooLongRetry:
    """main()'s too-long retry must give Spark a smaller diff, not no diff at all."""

    pr_data = {
        'number': 9, 'title': 'Big change', 'body': '', 'user': 'bob', 'changed_files': 1,
        'additions': 900, 'deletions': 0, 'files': [{'filename': 'app.py'}],
        'head': {'repo': {'full_name': 'o/r'}},
    }

    def test_retry_reviews_a_truncated_diff(self, capsys):
        from claudecode import github_action_audit as gaa

        prompts_sent = []

        def fake_post(url, **kwargs):
            prompts_sent.append(kwargs['json']['messages'][-1]['content'])
            response = Mock()
            if len(prompts_sent) == 1:
                response.status_code = 400
                response.text = "This model's maximum context length is 32768 tokens."
                return response
            response.status_code = 200
            response.text = ''
            response.json.return_value = {'choices': [
                {'message': {'content': '{"findings": []}'}, 'finish_reason': 'stop'}]}
            return response

        env = dict(SPARK_ENV, GITHUB_REPOSITORY='o/r', PR_NUMBER='9',
                   GITHUB_TOKEN='gh-token', LLM_PROVIDER='spark')
        with patch.dict(os.environ, env, clear=True), \
                patch.object(gaa, 'GitHubActionClient') as mock_github, \
                patch('claudecode.spark.requests.post', side_effect=fake_post):
            client = Mock()
            client.get_pr_data.return_value = self.pr_data
            client.get_pr_diff.return_value = 'diff --git a/app.py b/app.py\n' + 'x' * 200_000
            client._is_excluded.return_value = False
            mock_github.return_value = client
            with pytest.raises(SystemExit) as exc_info:
                gaa.main()

        assert exc_info.value.code == 0
        assert len(prompts_sent) == 2, 'the too-long response should trigger exactly one retry'
        assert 'diff truncated' in prompts_sent[1]
        assert len(prompts_sent[1]) < len(prompts_sent[0])
        # The retry must still contain code to review, not just a "diff omitted" note.
        assert 'x' * 10_000 in prompts_sent[1]
