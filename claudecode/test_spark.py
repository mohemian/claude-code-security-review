#!/usr/bin/env python3
"""Tests for the Spark provider: vLLM client, response parsing and finding validation."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from claudecode import filter_prompts
from claudecode.spark import (
    MAX_FIELD_CHARS,
    MAX_FINDINGS,
    PROMPT_TOO_LONG,
    SparkFilterClient,
    SparkProvider,
    VLLMClient,
    VLLMResponseError,
    sanitize_audit_results,
    strip_reasoning,
)

BASE_URL = 'http://dgx-spark:8000/v1'
MODEL = 'Qwen3-32B'


def make_client(**kwargs):
    kwargs.setdefault('base_url', BASE_URL)
    kwargs.setdefault('model', MODEL)
    return VLLMClient(**kwargs)


def chat_response(content, status=200, finish_reason='stop'):
    response = Mock()
    response.status_code = status
    response.json.return_value = {
        'choices': [{'message': {'role': 'assistant', 'content': content},
                     'finish_reason': finish_reason}]
    }
    response.text = content
    return response


class TestVLLMClientRequest:
    """Correct URL, model, request structure and auth header."""

    def test_endpoint_url(self):
        assert make_client().endpoint == 'http://dgx-spark:8000/v1/chat/completions'

    def test_trailing_slash_is_normalised(self):
        assert make_client(base_url=BASE_URL + '/').endpoint.endswith('/v1/chat/completions')

    @patch('claudecode.spark.requests.post')
    def test_request_structure(self, mock_post):
        mock_post.return_value = chat_response('{"findings": []}')
        make_client().chat([{'role': 'user', 'content': 'hello'}])

        url = mock_post.call_args[0][0]
        payload = mock_post.call_args[1]['json']
        assert url == 'http://dgx-spark:8000/v1/chat/completions'
        assert payload['model'] == MODEL
        assert payload['messages'] == [{'role': 'user', 'content': 'hello'}]
        assert payload['temperature'] == 0.0
        assert payload['stream'] is False
        assert payload['response_format'] == {'type': 'json_object'}
        assert mock_post.call_args[1]['timeout'] > 0

    @patch('claudecode.spark.requests.post')
    def test_no_auth_header_when_unauthenticated(self, mock_post):
        mock_post.return_value = chat_response('{}')
        make_client().chat([{'role': 'user', 'content': 'x'}])
        assert 'Authorization' not in mock_post.call_args[1]['headers']

    @patch('claudecode.spark.requests.post')
    def test_auth_header_when_api_key_configured(self, mock_post):
        mock_post.return_value = chat_response('{}')
        make_client(api_key='token-abc').chat([{'role': 'user', 'content': 'x'}])
        assert mock_post.call_args[1]['headers']['Authorization'] == 'Bearer token-abc'

    @patch('claudecode.spark.requests.post')
    def test_model_comes_from_environment_not_hardcoded(self, mock_post):
        mock_post.return_value = chat_response('{}')
        with patch.dict(os.environ, {'VLLM_BASE_URL': BASE_URL, 'VLLM_MODEL': 'Qwen3-Next-80B'}):
            VLLMClient.from_env().chat([{'role': 'user', 'content': 'x'}])
        assert mock_post.call_args[1]['json']['model'] == 'Qwen3-Next-80B'

    def test_from_env_reads_optional_api_key(self):
        with patch.dict(os.environ, {'VLLM_BASE_URL': BASE_URL, 'VLLM_MODEL': MODEL,
                                     'VLLM_API_KEY': 'secret'}, clear=True):
            assert VLLMClient.from_env().api_key == 'secret'
        with patch.dict(os.environ, {'VLLM_BASE_URL': BASE_URL, 'VLLM_MODEL': MODEL}, clear=True):
            assert VLLMClient.from_env().api_key is None


class TestVLLMClientResponses:
    """Response parsing, malformed payloads, HTTP errors, timeouts and retries."""

    @patch('claudecode.spark.requests.post')
    def test_parses_message_content(self, mock_post):
        mock_post.return_value = chat_response('{"findings": []}')
        success, content, error = make_client().chat([{'role': 'user', 'content': 'x'}])
        assert (success, content, error) == (True, '{"findings": []}', '')

    @patch('claudecode.spark.requests.post')
    def test_missing_choices_is_an_error(self, mock_post):
        response = Mock(status_code=200, text='{}')
        response.json.return_value = {'object': 'chat.completion'}
        mock_post.return_value = response
        success, _, error = make_client().chat([{'role': 'user', 'content': 'x'}])
        assert success is False
        assert 'no choices' in error

    @patch('claudecode.spark.requests.post')
    def test_empty_content_is_an_error(self, mock_post):
        mock_post.return_value = chat_response('   ')
        success, _, error = make_client().chat([{'role': 'user', 'content': 'x'}])
        assert success is False
        assert 'no content' in error

    @patch('claudecode.spark.requests.post')
    def test_non_json_body_is_an_error(self, mock_post):
        response = Mock(status_code=200, text='<html>proxy error</html>')
        response.json.side_effect = ValueError('not json')
        mock_post.return_value = response
        success, _, error = make_client().chat([{'role': 'user', 'content': 'x'}])
        assert success is False
        assert 'non-JSON body' in error

    @patch('claudecode.spark.requests.post')
    def test_error_field_in_body(self, mock_post):
        response = Mock(status_code=200, text='{}')
        response.json.return_value = {'error': {'message': 'model not loaded'}}
        mock_post.return_value = response
        success, _, error = make_client().chat([{'role': 'user', 'content': 'x'}])
        assert success is False
        assert 'model not loaded' in error

    @patch('claudecode.spark.time.sleep', lambda *_: None)
    @patch('claudecode.spark.requests.post')
    def test_client_error_is_not_retried(self, mock_post):
        mock_post.return_value = Mock(status_code=401, text='unauthorized')
        success, _, error = make_client().chat([{'role': 'user', 'content': 'x'}])
        assert success is False
        assert 'HTTP 401' in error
        assert mock_post.call_count == 1

    @patch('claudecode.spark.time.sleep', lambda *_: None)
    @patch('claudecode.spark.requests.post')
    def test_server_error_is_retried_then_reported(self, mock_post):
        mock_post.return_value = Mock(status_code=503, text='overloaded')
        success, _, error = make_client(max_retries=3).chat([{'role': 'user', 'content': 'x'}])
        assert success is False
        assert 'failed after 3 attempts' in error
        assert mock_post.call_count == 3

    @patch('claudecode.spark.time.sleep', lambda *_: None)
    @patch('claudecode.spark.requests.post')
    def test_retry_recovers(self, mock_post):
        mock_post.side_effect = [Mock(status_code=500, text='boom'),
                                 chat_response('{"findings": []}')]
        success, content, _ = make_client().chat([{'role': 'user', 'content': 'x'}])
        assert success is True
        assert content == '{"findings": []}'

    @patch('claudecode.spark.time.sleep', lambda *_: None)
    @patch('claudecode.spark.requests.post')
    def test_timeout_is_retried_and_reported(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout()
        success, _, error = make_client(max_retries=2, timeout_seconds=42).chat(
            [{'role': 'user', 'content': 'x'}])
        assert success is False
        assert 'timed out after 42s' in error
        assert mock_post.call_count == 2

    @patch('claudecode.spark.time.sleep', lambda *_: None)
    @patch('claudecode.spark.requests.post')
    def test_connection_error_is_retried(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError('refused')
        success, _, error = make_client(max_retries=2).chat([{'role': 'user', 'content': 'x'}])
        assert success is False
        assert 'vLLM request failed' in error

    @patch('claudecode.spark.requests.post')
    def test_context_length_error_maps_to_prompt_too_long(self, mock_post):
        mock_post.return_value = Mock(
            status_code=400,
            text='This model\'s maximum context length is 32768 tokens.')
        success, _, error = make_client().chat([{'role': 'user', 'content': 'x'}])
        assert success is False
        assert error == PROMPT_TOO_LONG

    @patch('claudecode.spark.requests.post')
    def test_falls_back_when_json_mode_is_unsupported(self, mock_post):
        mock_post.side_effect = [
            Mock(status_code=400, text='response_format is not supported'),
            chat_response('{"findings": []}'),
        ]
        client = make_client()
        success, content, _ = client.chat([{'role': 'user', 'content': 'x'}])
        assert success is True
        assert client.json_mode is False
        assert 'response_format' not in mock_post.call_args[1]['json']

    def test_unconfigured_client_fails_before_any_request(self):
        with patch('claudecode.spark.requests.post') as mock_post:
            success, _, error = VLLMClient(base_url='', model='').chat([])
        assert success is False
        assert 'VLLM_BASE_URL' in error
        mock_post.assert_not_called()


class TestReasoningStripping:
    def test_strips_think_block(self):
        assert strip_reasoning('<think>hmm {"a": 1}</think>{"findings": []}') == '{"findings": []}'

    def test_leaves_plain_text_alone(self):
        assert strip_reasoning('{"findings": []}') == '{"findings": []}'


class TestFindingValidation:
    """Model output is untrusted: valid reports pass, malformed ones raise."""

    valid = {
        'findings': [{
            'file': 'src/db.py', 'line': 42, 'severity': 'high',
            'category': 'sql_injection', 'description': 'Unparameterised query',
            'exploit_scenario': 'Attacker injects SQL', 'recommendation': 'Parameterise',
            'confidence': 0.95,
        }],
        'analysis_summary': {'files_reviewed': 3},
    }

    def test_valid_report(self):
        result = sanitize_audit_results(self.valid)
        finding = result['findings'][0]
        assert finding['file'] == 'src/db.py'
        assert finding['line'] == 42
        assert finding['severity'] == 'HIGH'
        assert finding['confidence'] == 0.95
        assert result['analysis_summary']['high_severity'] == 1
        assert result['analysis_summary']['review_completed'] is True

    def test_empty_findings_is_valid(self):
        assert sanitize_audit_results({'findings': []})['findings'] == []

    def test_string_line_number_is_coerced(self):
        report = json.loads(json.dumps(self.valid))
        report['findings'][0]['line'] = '42'
        assert sanitize_audit_results(report)['findings'][0]['line'] == 42

    def test_unknown_fields_are_dropped(self):
        report = json.loads(json.dumps(self.valid))
        report['findings'][0]['_filter_metadata'] = {'confidence_score': 10}
        report['findings'][0]['github_command'] = 'delete-repo'
        finding = sanitize_audit_results(report)['findings'][0]
        assert '_filter_metadata' not in finding
        assert 'github_command' not in finding

    def test_long_strings_are_clamped(self):
        report = json.loads(json.dumps(self.valid))
        report['findings'][0]['description'] = 'A' * 100_000
        assert len(sanitize_audit_results(report)['findings'][0]['description']) == MAX_FIELD_CHARS

    def test_not_an_object(self):
        with pytest.raises(VLLMResponseError, match='not a JSON object'):
            sanitize_audit_results(['findings'])

    def test_missing_findings_key(self):
        with pytest.raises(VLLMResponseError, match="missing the 'findings' key"):
            sanitize_audit_results({'analysis_summary': {}})

    def test_findings_not_a_list(self):
        with pytest.raises(VLLMResponseError, match='must be a list'):
            sanitize_audit_results({'findings': {'file': 'a.py'}})

    def test_finding_not_an_object(self):
        with pytest.raises(VLLMResponseError, match=r'findings\[0\] is not an object'):
            sanitize_audit_results({'findings': ['a vulnerability']})

    def test_missing_required_field(self):
        with pytest.raises(VLLMResponseError, match="missing required field 'description'"):
            sanitize_audit_results({'findings': [
                {'file': 'a.py', 'line': 1, 'severity': 'HIGH'}]})

    def test_invalid_severity(self):
        with pytest.raises(VLLMResponseError, match='CRITICAL'):
            sanitize_audit_results({'findings': [
                {'file': 'a.py', 'line': 1, 'severity': 'CRITICAL', 'description': 'x'}]})

    def test_non_numeric_line(self):
        with pytest.raises(VLLMResponseError, match='non-numeric line'):
            sanitize_audit_results({'findings': [
                {'file': 'a.py', 'line': 'somewhere', 'severity': 'HIGH', 'description': 'x'}]})

    @pytest.mark.parametrize('path', ['../../etc/passwd', 'a/../../b', '..'])
    def test_path_traversal_rejected(self, path):
        with pytest.raises(VLLMResponseError, match='unsafe file path'):
            sanitize_audit_results({'findings': [
                {'file': path, 'line': 1, 'severity': 'HIGH', 'description': 'x'}]})

    def test_absolute_path_is_normalised_to_repo_relative(self):
        result = sanitize_audit_results({'findings': [
            {'file': '/src/app.py', 'line': 1, 'severity': 'HIGH', 'description': 'x'}]})
        assert result['findings'][0]['file'] == 'src/app.py' 

    def test_absurd_finding_count_rejected(self):
        findings = [{'file': 'a.py', 'line': 1, 'severity': 'LOW', 'description': 'x'}] * (
            MAX_FINDINGS + 1)
        with pytest.raises(VLLMResponseError, match='above the'):
            sanitize_audit_results({'findings': findings})


class TestSparkAudit:
    """End-to-end provider behaviour with the HTTP layer mocked."""

    def provider(self):
        return SparkProvider(client=make_client())

    @patch('claudecode.spark.requests.post')
    def test_successful_audit(self, mock_post):
        mock_post.return_value = chat_response(json.dumps(TestFindingValidation.valid))
        success, error, results = self.provider().run_security_audit(Path('.'), 'prompt')
        assert success is True
        assert error == ''
        assert results['findings'][0]['severity'] == 'HIGH'

    @patch('claudecode.spark.requests.post')
    def test_markdown_fenced_response_is_parsed(self, mock_post):
        fenced = '```json\n' + json.dumps({'findings': []}) + '\n```'
        mock_post.return_value = chat_response(fenced)
        success, _, results = self.provider().run_security_audit(Path('.'), 'prompt')
        assert success is True
        assert results['findings'] == []

    @patch('claudecode.spark.requests.post')
    def test_prose_wrapped_response_is_parsed(self, mock_post):
        mock_post.return_value = chat_response(
            'Sure! Here is the report:\n{"findings": []}\nLet me know if you need more.')
        success, _, results = self.provider().run_security_audit(Path('.'), 'prompt')
        assert success is True

    @patch('claudecode.spark.requests.post')
    def test_thinking_block_is_stripped_before_parsing(self, mock_post):
        mock_post.return_value = chat_response(
            '<think>Maybe {"findings": [{"file": "x"}]} is wrong</think>{"findings": []}')
        success, _, results = self.provider().run_security_audit(Path('.'), 'prompt')
        assert success is True
        assert results['findings'] == []

    @patch('claudecode.spark.requests.post')
    def test_unparseable_response_is_an_error(self, mock_post):
        mock_post.return_value = chat_response('I could not complete this review.')
        success, error, results = self.provider().run_security_audit(Path('.'), 'prompt')
        assert success is False
        assert 'Could not parse JSON' in error
        assert results == {}

    @patch('claudecode.spark.requests.post')
    def test_malformed_findings_are_not_silently_accepted(self, mock_post):
        mock_post.return_value = chat_response(
            '{"findings": [{"file": "a.py", "line": 1, "severity": "CATASTROPHIC", '
            '"description": "x"}]}')
        success, error, _ = self.provider().run_security_audit(Path('.'), 'prompt')
        assert success is False
        assert 'Invalid security report from model' in error

    @patch('claudecode.spark.requests.post')
    def test_prompt_too_long_sentinel_propagates(self, mock_post):
        mock_post.return_value = Mock(status_code=400, text='maximum context length exceeded')
        success, error, _ = self.provider().run_security_audit(Path('.'), 'prompt')
        assert success is False
        assert error == PROMPT_TOO_LONG


class TestSparkFilterClient:
    """False-positive filtering runs through vLLM with the shared prompts."""

    def client(self):
        return SparkFilterClient(make_client())

    finding = {'file': 'a.py', 'line': 1, 'severity': 'HIGH', 'description': 'SQLi'}

    @patch('claudecode.spark.requests.post')
    def test_keeps_finding(self, mock_post):
        mock_post.return_value = chat_response(json.dumps(
            {'confidence_score': 9, 'keep_finding': True, 'justification': 'real'}))
        success, result, error = self.client().analyze_single_finding(self.finding)
        assert success is True
        assert result['keep_finding'] is True
        assert error == ''

    @patch('claudecode.spark.requests.post')
    def test_excludes_finding(self, mock_post):
        mock_post.return_value = chat_response(json.dumps(
            {'confidence_score': 2, 'keep_finding': False, 'exclusion_reason': 'test file'}))
        success, result, _ = self.client().analyze_single_finding(self.finding)
        assert success is True
        assert result['keep_finding'] is False

    @patch('claudecode.spark.requests.post')
    def test_uses_the_shared_filter_prompts(self, mock_post):
        mock_post.return_value = chat_response('{"keep_finding": true}')
        self.client().analyze_single_finding(self.finding, {'repo_name': 'o/r', 'pr_number': 5})
        messages = mock_post.call_args[1]['json']['messages']
        assert messages[0]['role'] == 'system'
        assert 'false positives' in messages[0]['content']
        assert 'HARD EXCLUSIONS' in messages[1]['content']
        assert 'o/r' in messages[1]['content']

    @patch('claudecode.spark.requests.post')
    def test_missing_verdict_is_an_error_not_a_drop(self, mock_post):
        mock_post.return_value = chat_response('{"justification": "seems fine"}')
        success, _, error = self.client().analyze_single_finding(self.finding)
        assert success is False
        assert 'keep_finding' in error

    @patch('claudecode.spark.requests.post')
    def test_unparseable_verdict_is_an_error(self, mock_post):
        mock_post.return_value = chat_response('probably fine')
        success, _, error = self.client().analyze_single_finding(self.finding)
        assert success is False
        assert 'Failed to parse JSON' in error


class TestFilterFileReadContainment:
    """Model-supplied paths must not pull runner files into the filtering prompt."""

    def test_reads_inside_the_repo(self, tmp_path):
        (tmp_path / 'app.py').write_text('print("hi")')
        with patch.dict(os.environ, {'REPO_PATH': str(tmp_path)}):
            success, content, _ = filter_prompts.read_repo_file('app.py')
        assert success is True
        assert 'print("hi")' in content

    @pytest.mark.parametrize('path', ['../secret.txt', '../../etc/passwd'])
    def test_refuses_to_escape_the_repo(self, tmp_path, path):
        repo = tmp_path / 'repo'
        repo.mkdir()
        (tmp_path / 'secret.txt').write_text('ANTHROPIC_API_KEY=sk-real')
        with patch.dict(os.environ, {'REPO_PATH': str(repo)}):
            success, content, error = filter_prompts.read_repo_file(path)
        assert success is False
        assert content == ''
        assert 'outside the repository' in error

    def test_refuses_absolute_paths_outside_the_repo(self, tmp_path):
        secret = tmp_path / 'secret.txt'
        secret.write_text('token')
        repo = tmp_path / 'repo'
        repo.mkdir()
        with patch.dict(os.environ, {'REPO_PATH': str(repo)}):
            success, _, error = filter_prompts.read_repo_file(str(secret))
        assert success is False
        assert 'outside the repository' in error
