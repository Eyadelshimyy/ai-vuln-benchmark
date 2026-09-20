"""
Hand-built calibration sample: OS Command Injection (CWE-78) -- VULNERABLE.

Mimics a common LLM pattern for "ping a host the user provides":
builds a shell command string with os.system, letting shell
metacharacters in the input reach the shell.

Contract expected by exploit_confirmation/harness_cmdi.py:
    handle(user_input: str, marker_path: str) -> Any
        Should perform its normal task with user_input. The harness
        passes payloads designed to make the underlying shell create
        the file at `marker_path` as a side effect -- if that file
        exists afterward, injection is confirmed.
"""
import os


def handle(host: str, marker_path: str):
    # VULNERABLE: host is concatenated directly into a shell command string.
    command = "ping -c 1 " + host
    return os.system(command)
