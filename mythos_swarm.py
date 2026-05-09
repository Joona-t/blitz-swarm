#!/usr/bin/env python3
"""Mythos Swarm — convenience entrypoint.

Equivalent to: python orchestrator.py --mode mythos "task spec"

    python mythos_swarm.py "Implement & verify a binary search with loop invariants"
    python mythos_swarm.py "task" --cost-ceiling 2.0 --max-replans 2
    python mythos_swarm.py "task" --dry-run

All CLI flags pass through to orchestrator.py.
"""

import os
import sys


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    orch = os.path.join(here, "orchestrator.py")
    args = ["python3", orch, "--mode", "mythos"] + sys.argv[1:]
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
