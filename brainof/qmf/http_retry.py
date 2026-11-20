"""Retry transient metadata failures without treating them as missing participants."""
import time
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def is_retryable_network_error(exc):
    if isinstance(exc, HTTPError):
        return exc.code in (408, 429, 500, 502, 503, 504)
    if isinstance(exc, (URLError, TimeoutError, ConnectionError)):
        return True
    return (isinstance(exc, subprocess.CalledProcessError) and
            isinstance(exc.cmd, (list, tuple)) and exc.cmd[0] == 'curl' and
            exc.returncode in (5, 6, 7, 18, 28, 35, 52, 55, 56, 92))


def urlopen_retry(request, timeout=60):
    for attempt in range(6):
        try:
            return urlopen(request, timeout=timeout)
        except (URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 5:
                raise
            print(f'Metadata connection interrupted; retry {attempt+1}/5 in 30 seconds', flush=True)
            time.sleep(30)


if __name__ == '__main__':
    from unittest.mock import patch
    import sys
    module = sys.modules[__name__]
    with patch.object(module, 'urlopen', side_effect=[URLError('transient'), 'ok']) as get, \
         patch.object(time, 'sleep'):
        assert urlopen_retry('https://example.org') == 'ok'
        assert get.call_count == 2
    for code, expected_calls in [(404, 1), (503, 6)]:
        with patch.object(module, 'urlopen', side_effect=HTTPError('https://example.org', code, '', {}, None)) as get, \
             patch.object(time, 'sleep'):
            try:
                urlopen_retry('https://example.org')
            except HTTPError as exc:
                assert exc.code == code
            else:
                raise AssertionError('HTTP error was swallowed')
            assert get.call_count == expected_calls
    print('Metadata retries: recovery, bounded retries and immediate 404 propagation passed')
    assert is_retryable_network_error(URLError('TLS interrupted'))
    assert is_retryable_network_error(subprocess.CalledProcessError(35, ['curl']))
    assert not is_retryable_network_error(subprocess.CalledProcessError(23, ['curl']))
    assert not is_retryable_network_error(subprocess.CalledProcessError(35, ['python']))
    assert not is_retryable_network_error(ValueError('invalid data'))
    assert not is_retryable_network_error(HTTPError('x', 404, '', {}, None))
    print('Network restart classification: transient transport only, data/write errors excluded')
