#!/usr/bin/env python3
"""A stand-in for vLLM, for testing the Spark provider without a DGX Spark.

Speaks just enough of the OpenAI chat-completions API to exercise the real HTTP
path: it echoes back canned findings for an audit prompt and a keep/drop verdict
for a filtering prompt.

    python3 scripts/fake-vllm-server.py --port 8000
    # then point the action at http://localhost:8000/v1

Modes (--mode) let you rehearse the failure paths:
    findings   valid report with one HIGH finding (default)
    empty      valid report, no findings
    malformed  a finding with a bogus severity -- the run must fail, not drop it
    prose      no JSON at all -- parse error
    toolong    HTTP 400 context-length -- must trigger the truncated-diff retry
    flaky      HTTP 500 on the first call, then valid -- must retry
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

FINDING = {
    'file': 'app.py', 'line': 1, 'severity': 'HIGH', 'category': 'sql_injection',
    'description': 'User input is interpolated into a SQL query',
    'exploit_scenario': "Attacker submits 1' OR '1'='1 as the search parameter",
    'recommendation': 'Use parameterised queries', 'confidence': 0.95,
}
VERDICT = {'original_severity': 'HIGH', 'confidence_score': 9, 'keep_finding': True,
           'exclusion_reason': None, 'justification': 'Concrete injection path'}

BODIES = {
    'findings': {'findings': [FINDING], 'analysis_summary': {'files_reviewed': 1}},
    'empty': {'findings': [], 'analysis_summary': {'files_reviewed': 1}},
    'malformed': {'findings': [dict(FINDING, severity='CATASTROPHIC')]},
}


def _first_diff_location(prompt):
    """Find a (file, line) that is genuinely inside the PR diff.

    GitHub only accepts a review comment on a line the diff touches, so anchoring the
    canned finding to the first hunk keeps the PR-commenting step honest.
    """
    diff = re.search(r'^diff --git a/(\S+) b/\S+(.*?)(?=^diff --git |\Z)',
                     prompt, re.MULTILINE | re.DOTALL)
    if diff:
        hunk = re.search(r'^@@ -\d+(?:,\d+)? \+(\d+)', diff.group(2), re.MULTILINE)
        if hunk:
            return diff.group(1), int(hunk.group(1))

    listed = re.search(r'^Files modified:\n- (.+)$', prompt, re.MULTILINE)
    return (listed.group(1).strip() if listed else 'app.py'), 1


class Handler(BaseHTTPRequestHandler):
    mode = 'findings'
    calls = 0

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
        prompt = request['messages'][-1]['content']
        is_filter = 'Finding to analyze' in prompt
        Handler.calls += 1

        print(f'  -> {self.path} call={Handler.calls} model={request.get("model")!r} '
              f'kind={"filter" if is_filter else "audit"} prompt={len(prompt)} chars '
              f'auth={"yes" if self.headers.get("Authorization") else "no"} '
              f'json_mode={"response_format" in request}')

        if not is_filter and self.mode == 'toolong':
            return self._send(400, "This model's maximum context length is 32768 tokens.")
        if not is_filter and self.mode == 'flaky' and Handler.calls == 1:
            return self._send(500, 'engine is warming up')

        if is_filter:
            content = json.dumps(VERDICT)
        elif self.mode == 'prose':
            content = 'I was unable to complete this security review.'
        else:
            body = json.loads(json.dumps(BODIES.get(self.mode, BODIES['findings'])))
            # Report against a file that is really in the PR, so the finding survives
            # validation and GitHub accepts the review comment.
            changed_file, changed_line = _first_diff_location(prompt)
            for finding in body['findings']:
                finding['file'] = changed_file
                finding['line'] = changed_line
            content = json.dumps(body)

        self._send(200, json.dumps({
            'model': request.get('model'),
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': content}}],
        }))

    def log_message(self, *args):
        pass  # the do_POST print above is the log we want

    def _send(self, status, body):
        payload = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--mode', default='findings', choices=[
        'findings', 'empty', 'malformed', 'prose', 'toolong', 'flaky'])
    args = parser.parse_args()

    Handler.mode = args.mode
    print(f'fake vLLM ({args.mode}) on http://localhost:{args.port}/v1')
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
