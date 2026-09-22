"""
Regression guard for Review 1 condition #3 (no-model-inference). If any
future change to app/pipeline.py (or anything it imports) tries to open a
network socket during correct_text(), this test fails loudly rather than
silently degrading into something that calls an external API.

For the full evidence packet (static import scan + this same network
check + GPU check + CPU/RAM profiling), see
scripts/no_model_inference_proof.py and reports/no_model_inference_proof.txt.
"""
import socket

import pytest

from app.pipeline import correct_text
from tests.batch_cases import BATCH_CASES


class _NetworkAttemptBlocked(Exception):
    pass


def _blocked_socket(*args, **kwargs):
    raise _NetworkAttemptBlocked("network socket opened during correct_text()")


def test_no_network_calls_during_correction():
    original_socket = socket.socket
    socket.socket = _blocked_socket
    try:
        for text, _ in BATCH_CASES:
            correct_text(text)
    except _NetworkAttemptBlocked as e:
        pytest.fail(str(e))
    finally:
        socket.socket = original_socket
