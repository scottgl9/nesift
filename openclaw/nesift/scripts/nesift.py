#!/usr/bin/env python3
"""OpenClaw bridge: thin wrapper that forwards to the installed `nesift` CLI.

OpenClaw skills can shell out to this file before nesift is published to
PyPI. Once `pip install nesift` is available, prefer the system CLI.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    exe = shutil.which("nesift")
    if exe is None:
        sys.stderr.write(
            "nesift CLI not found in PATH. Install with `pip install nesift` "
            "or clone https://github.com/scottgl9/nesift and `pip install -e .`.\n"
        )
        return 127
    return subprocess.call([exe, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
