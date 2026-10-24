from __future__ import annotations

import json
import stat
import urllib.request
from pathlib import Path

import pytest
from arxiv_summarizer import OpenRouterSummarizer


@pytest.mark.parametrize(('response', 'reason'), [
    ('not JSON', 'response is not JSON'),
    ('```json\n[]\n```', 'response is not JSON'),
    ('[{"id":"wrong","summary":"Text"}]', 'response IDs do not match the requested IDs'),
    ('[{"id":"paper","summary":"cut off', 'response is not JSON'),
])
def test_rejected_response_private_diagnostics(tmp_path: Path, monkeypatch, capsys, response, reason):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({'model': 'actual/model', 'choices': [{
                'message': {'content': response}, 'finish_reason': 'length',
            }], 'unrelated_secret': 'must-not-save'}).encode()

    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: Response())
    directory = tmp_path / 'diagnostics'
    backend = OpenRouterSummarizer(model='openrouter/free', api_key='secret-api-key', diagnostics_dir=directory)
    result = backend.summarize_items([{'id': 'paper', 'text': 'Abstract'}])[0]
    assert result['summary_metadata'] == dict(requested_model='openrouter/free', returned_model='actual/model', finish_reason='length')
    assert result['error'] == f'Invalid summary response: {reason}'
    files = list(directory.glob('*.json'))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record == dict(requested_ids=['paper'], requested_model='openrouter/free',
                          returned_model='actual/model', finish_reason='length',
                          rejection_reason=reason, response=response)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    assert 'secret-api-key' not in files[0].read_text()
    assert 'must-not-save' not in files[0].read_text()
    assert capsys.readouterr() == ('', '')


@pytest.mark.parametrize('success', [True, False])
def test_completion_metadata_in_paper_output(tmp_path, monkeypatch, success):
    from arxiv_summarizer.client import CompletionText
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    from scripts.run_test_pipeline import run_pipeline

    class Client:
        def chat(self, *args, **kwargs):
            text = '[{"id":"paper","summary":"Summary."}]' if success else 'bad JSON'
            return CompletionText(text, 'actual/model', 'stop')

    config = tmp_path / 'interests.yaml'
    config.write_text('- name: quantum\n  weight: 1\n  keywords: [quantum]\n')
    papers = [{'id': 'paper', 'title': 'quantum', 'abstract': 'quantum abstract'}]
    backend = OpenRouterSummarizer(client=Client(), model='openrouter/free')
    output = run_pipeline(papers, config, backend)
    assert output[0]['summary_metadata'] == dict(requested_model='openrouter/free', returned_model='actual/model', finish_reason='stop')
    assert output[0]['summary_status'] == ('ok' if success else 'error')


def test_diagnostics_default_off_and_request_errors_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Client:
        def chat(self, *a, **kw):
            return 'private response'

    result = OpenRouterSummarizer(client=Client()).summarize('Abstract')
    assert result['error'] == 'Invalid summary response: response is not JSON'
    assert not list(tmp_path.iterdir())

    class BrokenClient:
        def chat(self, *a, **kw):
            raise ValueError('sensitive API exception')

    result = OpenRouterSummarizer(client=BrokenClient(), diagnostics_dir=tmp_path / 'diagnostics').summarize('Abstract')
    assert result['error'] == 'ValueError: sensitive API exception'
    files = list((tmp_path / 'diagnostics').glob('request-error-*.json'))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record['exception_message'] == 'sensitive API exception'
    assert record['exception_type'] == 'ValueError'
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600


def test_http_error_body_preserved(tmp_path):
    import io
    import urllib.error

    body = b'{"error":{"message":"Provider unavailable","code":503}}'

    class Client:
        def chat(self, *args, **kwargs):
            raise urllib.error.HTTPError('https://example.test', 503, 'Unavailable', None, io.BytesIO(body))

    result = OpenRouterSummarizer(client=Client(), diagnostics_dir=tmp_path / 'diagnostics').summarize('Abstract')
    assert 'Provider unavailable' in result['error']
    record = json.loads(next((tmp_path / 'diagnostics').glob('request-error-*.json')).read_text())
    assert record['response_body'] == body.decode()
    assert record['http_status'] == 503


def test_null_content_error_retains_api_response(monkeypatch):
    from arxiv_summarizer.client import LLMClient

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b'{"choices":[{"message":{"content":null},"finish_reason":"length"}],"model":"actual/model"}'

    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: Response())
    with pytest.raises(ValueError, match='actual/model'):
        LLMClient('https://example.test', 'test-key', 'test-model').chat([])
