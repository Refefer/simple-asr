"""Utility functions for file naming and keyboard handling."""

import os
import re
import sys
import termios
import tty
from pathlib import Path
from datetime import datetime


def get_next_filename(output_dir: Path, prefix: str) -> Path:
    """Generate the next available filename with version increment.

    Returns path like: {output_dir}/<datetime>-{prefix}-{increment:03d}.txt
    """
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pattern = re.compile(rf"^{re.escape(dt_str)}-{re.escape(prefix)}-(\d{{3}})\.txt$")
    max_version = 0

    if output_dir.exists():
        for entry in output_dir.iterdir():
            match = pattern.match(entry.name)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)

    next_version = max_version + 1
    return output_dir / f"{dt_str}-{prefix}-{next_version:03d}.txt"


class KeyboardHandler:
    """Non-blocking keyboard input handler for Linux."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args):
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key(self) -> str | None:
        """Get a keypress if available, non-blocking."""
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # Escape sequence
                # Check for more characters (arrow keys, etc.)
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    sys.stdin.read(2)  # Consume rest of escape sequence
                return 'ESC'
            elif ch == '\n' or ch == '\r':
                return 'ENTER'
            elif ch == '\x03':  # Ctrl-C
                return 'CTRL-C'
            return ch
        return None


def clear_line():
    """Clear the current console line."""
    sys.stdout.write('\r\033[K')
    sys.stdout.flush()


def print_status(message: str, end: str = '\n'):
    """Print a status message."""
    clear_line()
    print(message, end=end, flush=True)
