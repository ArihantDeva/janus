"""Installed Muse entrypoint: encoder verdict from the owned Amarel allocation.

A transport failure prints one sanitized line to stderr and emits no stdout, so
the tool turn proceeds without fabricated advice. Diagnostics never reach stdout.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import muse_hook as H


def main():
    try:
        return H.main()
    except Exception as error:  # never break the user's tool turn
        print(f"Tool failure classifier unavailable: {type(error).__name__}",
              file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
