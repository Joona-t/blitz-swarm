"""Human labels for calibrating the judge (Phase 1, WP9).

    python -m bench.label --runs bench/runs/<S> bench/runs/<B0> ... [--ref B1] [--n 30]

Shows one blinded pair at a time (arm vs reference, sides randomised),
asks which report is better on each criterion, and appends to
labels.jsonl. Resumable: pairs already labelled are skipped. Labels are
stored un-randomised as x (the arm) / y (the reference) / tie under the
same pair key the judge cache uses, so `bench.compare --labels` can line
them up and report Cohen's kappa.

Keys while reading: the two reports are written next to the labels file as
PAIR_A.md and PAIR_B.md and opened in $PAGER (default `less`). Answer each
criterion with a / b / t; q quits and keeps what was labelled.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

from bench.judge import CRITERIA, _pair_key

QUESTIONS = {
    "accuracy": "fewer wrong or unsupported claims",
    "completeness": "covers more of what the task asks",
    "specificity": "more concrete (mechanisms, numbers, conditions)",
    "consistency": "contradicts itself less",
}


def build_pairs(arms: dict[str, dict], ref: str, n: int, seed: int = 42) -> list[dict]:
    """Up to n (arm, prompt) pairs vs the reference, spread across arms."""
    rng = random.Random(seed)
    pool = []
    for arm, scores in sorted(arms.items()):
        if arm == ref:
            continue
        for pid in sorted(set(scores) & set(arms[ref])):
            x, y = scores[pid], arms[ref][pid]
            pool.append({"arm": arm, "prompt_id": pid, "task": x["task"], "x": x["text"], "y": y["text"],
                         "pair_key": _pair_key(x["task"], x["text"], y["text"])})
    rng.shuffle(pool)
    picked = pool[:n]
    for p in picked:
        p["x_is_a"] = rng.random() < 0.5
    return picked


def _pager_show(path_a: Path, path_b: Path) -> None:
    pager = os.environ.get("PAGER", "less")
    for p in (path_a, path_b):
        subprocess.run([pager, str(p)], check=False)


def label_pairs(pairs: list[dict], labels_path: Path, *, input_fn=input, show_fn=_pager_show, out=print) -> int:
    """Interactive loop. Returns the number of pairs labelled in this sitting."""
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line).get("pair_key"))
    todo = [p for p in pairs if p["pair_key"] not in done]
    out(f"{len(done)} labelled already, {len(todo)} to go. a / b / t(ie) per question, q to stop.")
    labelled = 0
    for i, p in enumerate(todo, 1):
        a_text, b_text = (p["x"], p["y"]) if p["x_is_a"] else (p["y"], p["x"])
        pa, pb = labels_path.parent / "PAIR_A.md", labels_path.parent / "PAIR_B.md"
        pa.write_text(f"# REPORT A\n\nTask: {p['task']}\n\n---\n\n{a_text}", encoding="utf-8")
        pb.write_text(f"# REPORT B\n\nTask: {p['task']}\n\n---\n\n{b_text}", encoding="utf-8")
        out(f"\n[{i}/{len(todo)}] Task: {p['task']}\n  A: {pa} ({len(a_text)} chars)\n  B: {pb} ({len(b_text)} chars)")
        show_fn(pa, pb)
        criteria: dict[str, str] = {}
        quit_now = False
        for c in CRITERIA:
            while True:
                ans = input_fn(f"  {c} ({QUESTIONS[c]}) [a/b/t/q]: ").strip().lower()[:1]
                if ans in ("a", "b", "t", "q"):
                    break
            if ans == "q":
                quit_now = True
                break
            side = {"a": "x" if p["x_is_a"] else "y", "b": "y" if p["x_is_a"] else "x", "t": "tie"}[ans]
            criteria[c] = side
        if quit_now:
            break
        with labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"pair_key": p["pair_key"], "arm": p["arm"], "prompt_id": p["prompt_id"],
                                "criteria": criteria}) + "\n")
        labelled += 1
    out(f"labelled {labelled} this sitting -> {labels_path}")
    return labelled


def main():
    import argparse
    from bench import load_slate
    from bench.compare import load_run, score_run
    ap = argparse.ArgumentParser(description="Label blinded report pairs for judge calibration")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", default="B1")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slate", default=str(Path(__file__).parent / "slate_v1.toml"))
    ap.add_argument("--labels", default=str(Path(__file__).parent / "comparisons" / "labels" / "labels.jsonl"))
    args = ap.parse_args()
    slate = load_slate(args.slate)
    arms: dict[str, dict] = {}
    for d in args.runs:
        run = load_run(Path(d))
        arms.setdefault(run["arm"], {}).update(score_run(run, slate, citations="off"))
    label_pairs(build_pairs(arms, args.ref, args.n, args.seed), Path(args.labels))


if __name__ == "__main__":
    main()
