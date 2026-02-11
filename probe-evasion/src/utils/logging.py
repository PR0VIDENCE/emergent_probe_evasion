"""Dual logging to console + file for long-running GPU scripts."""

import os
import sys
from datetime import datetime


class TeeStream:
    """Write to both a file and the original stream."""

    def __init__(self, file_handle, original_stream):
        self.file = file_handle
        self.stream = original_stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()


def setup_logging(log_dir: str, script_name: str) -> str:
    """
    Set up dual output: everything printed to console also goes to a log file.

    Creates a timestamped log file in log_dir. Captures both stdout and stderr.
    Works with print(), tqdm, warnings — no code changes needed.

    Args:
        log_dir: Directory to write log files.
        script_name: Name prefix for the log file.

    Returns:
        Path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    log_file = open(log_path, "w")
    sys.stdout = TeeStream(log_file, sys.__stdout__)
    sys.stderr = TeeStream(log_file, sys.__stderr__)

    print(f"Logging to: {log_path}")
    return log_path
