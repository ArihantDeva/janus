"""The retry_wait relabel must fire on whole-error transients and NOT on content.

A 4000-char build log that merely contains the word "timeout" is a debug row
whose content mentions timeouts; "Command timed out after 120 seconds" as the
whole error is the environment being slow. The rule distinguishes them by
position (head-of-error) and by ruiners (traceback/diff text means code, not
weather). Both directions pinned here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from relabel_retry_wait import is_transient  # noqa: E402


def test_whole_error_timeout_is_transient():
    assert is_transient("Command timed out after 120 seconds")
    assert is_transient("Command timed out after 7200 seconds")


def test_rate_limit_head_is_transient():
    assert is_transient("HTTP 429: too many requests, retry later")
    assert is_transient("rate limit exceeded; backoff 60s")


def test_ssh_timeout_is_transient():
    assert is_transient(
        "ssh: connect to host 10.75.227.201 port 22: Operation timed out")


def test_mid_log_mention_is_not_transient():
    assert not is_transient(
        "build ok; caches transient empty results in process (268ms)\n"
        + "log line\n" * 40)


def test_traceback_quoting_timeout_is_not_transient():
    assert not is_transient(
        "Traceback (most recent call last):\n"
        '  File "x.py", line 3, in run\n'
        "    stdout, stderr = process.communicate(input, timeout=timeout)\n"
        "TimeoutError: timed out")


def test_missing_file_with_timeout_word_is_not_transient():
    assert not is_transient(
        "sed: src/backend/rateLimit.js: No such file or directory")
