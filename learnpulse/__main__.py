"""Command line: python -m learnpulse {build,serve,up}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as C
from .store import DEFAULT_DB


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="learnpulse", description="LearnPulse prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate the synthetic school, run the weekly pipeline, write the evaluation report")
    b.add_argument("--db", type=Path, default=DEFAULT_DB)
    b.add_argument("--seed", type=int, default=C.SEED)
    for name in ("serve", "up"):
        s = sub.add_parser(name, help="start the API and teacher panel" if name == "serve" else "build if needed, then serve")
        s.add_argument("--db", type=Path, default=DEFAULT_DB)
        s.add_argument("--host", default="127.0.0.1")
        s.add_argument("--port", type=int, default=8000)
        s.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args(argv)

    if a.cmd == "build" or (a.cmd == "up" and not a.db.exists()):
        from .pipeline import build
        report = build(a.db, a.seed)
        print(json.dumps({"metrics": report["metrics"], "timings_seconds": report["timings_seconds"]}, indent=2))
        print("Report: reports/evaluation.md")
    if a.cmd in ("serve", "up"):
        import uvicorn

        from .api import create_app
        print(f"LearnPulse teacher panel: http://{a.host}:{a.port}/")
        uvicorn.run(create_app(a.db), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
