"""Cross-arm comparison (Phase 1, WP8-9).

    python -m bench.compare --runs bench/runs/<S> bench/runs/<B0> ... \
        [--ref B1] [--citations live|offline|off] [--judge] [--labels labels.jsonl] [--out DIR]

Every arm is RE-SCORED here from its report text with the same code and the
same settings (citation check, coverage, length; novelty against nothing),
so no arm benefits from having been scored by an older runner. Then, per arm
vs the reference arm on the prompts both completed: mean difference,
bootstrap CI, paired t, Cohen's d_z; optional blind pairwise judging (M2);
optional agreement between the judge and human labels (Cohen's kappa).

Prompts missing from an arm are listed, never silently dropped. The report
states n and the smallest effect that n could detect.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from bench import load_slate
from bench.citations import check_report
from bench.detectors import research_quality_composite
from bench.stats import bootstrap_ci, cohens_d, paired_t_test, t_two_sided_p

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = ("composite_v2", "citations_ratio", "coverage", "composite_v1")


def _arm_of(run_dir: Path) -> str:
    try:
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("arm") or "S"
    except (OSError, json.JSONDecodeError):
        return "S"


def _report_path(run_dir: Path, row: dict) -> Path | None:
    """Committed copy first (<run>/reports/<prompt_id>.md or <key>.md), then
    the path the runner recorded (repo-relative or ~-collapsed)."""
    from bench.arms import prompt_key
    candidates = [run_dir / "reports" / f"{row.get('prompt_id')}.md"]
    recorded = str(row.get("output_md_path") or "")
    if recorded:
        candidates.append(run_dir / "reports" / Path(recorded).name)
        candidates.append(Path(recorded).expanduser() if recorded.startswith("~") else REPO_ROOT / recorded)
        candidates.append(REPO_ROOT / "output" / Path(recorded).name)
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_run(run_dir: Path) -> dict:
    """{arm, run_dir, rows: {prompt_id: row}, skipped: {prompt_id: reason}}."""
    run_dir = Path(run_dir)
    rows, skipped = {}, {}
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid = row.get("prompt_id")
        if row.get("error") or row.get("timeout"):
            skipped[pid] = row.get("error") or "timeout"
        else:
            rows[pid] = row
    return {"arm": _arm_of(run_dir), "run_dir": run_dir, "rows": rows, "skipped": skipped}


def score_run(run: dict, slate, *, citations: str = "offline", check=check_report) -> dict:
    """Re-score every completed prompt of a run from its report text."""
    scored = {}
    for pid, row in run["rows"].items():
        path = _report_path(run["run_dir"], row)
        if path is None:
            run["skipped"][pid] = "report file not found"
            continue
        text = path.read_text(encoding="utf-8")
        try:
            prompt = slate.by_id(pid)
        except KeyError:
            run["skipped"][pid] = "prompt not in slate"
            continue
        cc = None
        if citations in ("live", "offline"):
            cc = check(text, offline=citations == "offline")
        fc = research_quality_composite(text, {"expected_subtopics": list(prompt.expected_coverage)},
                                        citation_check=cc)
        scored[pid] = {
            "tier": prompt.tier, "task": prompt.text, "text": text, "report": str(path),
            "composite_v1": fc["composite"], "composite_v2": fc["composite_v2"], "coverage": fc["coverage"],
            "citations_ratio": cc["ratio"] if cc else None,
            "n_cited": cc["n_cited"] if cc else None, "resolved": cc["resolved"] if cc else None,
            "not_found": cc["not_found"] if cc else None, "title_mismatch": cc["title_mismatch"] if cc else None,
            "unverifiable": cc["unverifiable"] if cc else None,
            "length_chars": fc["length_chars"],
            "output_tokens": row.get("output_tokens", 0),
            "generation_output_tokens": row.get("generation_output_tokens") or row.get("output_tokens", 0),
            "calls": row.get("calls", 0), "elapsed_s": row.get("elapsed_s", 0.0),
            "cost_usd": row.get("cost_usd", 0.0), "cost_incomplete": row.get("cost_incomplete", False),
            "beta": (row.get("beta") or {}).get("beta"),
        }
    return scored


def _mean_ci(values: list[float], seed: int = 42) -> dict:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"n": 0, "mean": None, "ci95": None}
    lo, hi = bootstrap_ci(vals, ci=95.0, n_resamples=2000, seed=seed) if len(vals) > 1 else (vals[0], vals[0])
    return {"n": len(vals), "mean": round(statistics.mean(vals), 4), "ci95": [round(lo, 4), round(hi, 4)]}


def paired(arm_scores: dict, ref_scores: dict, metric: str, seed: int = 42) -> dict:
    """Arm minus reference on the prompts both have a value for."""
    pids = sorted(p for p in arm_scores if p in ref_scores
                  and isinstance(arm_scores[p].get(metric), (int, float))
                  and isinstance(ref_scores[p].get(metric), (int, float)))
    a = [arm_scores[p][metric] for p in pids]
    b = [ref_scores[p][metric] for p in pids]
    if len(pids) < 2:
        return {"n": len(pids), "prompts": pids}
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs, ci=95.0, n_resamples=2000, seed=seed)
    t, p, df = paired_t_test(a, b)
    return {
        "n": len(pids), "prompts": pids, "mean_delta": round(statistics.mean(diffs), 4),
        "ci95": [round(lo, 4), round(hi, 4)], "t": round(t, 3), "df": df, "p": round(p, 4),
        "d_z": round(cohens_d(a, b), 3),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def min_detectable_dz(n: int, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Approximate smallest paired effect detectable at this n: (t_crit + z_power) / sqrt(n)."""
    if n < 3:
        return None
    lo, hi = 0.0, 50.0
    for _ in range(80):                      # bisection for the two-sided critical t
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, n - 1) > alpha:
            lo = mid
        else:
            hi = mid
    z_power = 0.8416 if abs(power - 0.8) < 1e-9 else statistics.NormalDist().inv_cdf(power)
    return round((hi + z_power) / math.sqrt(n), 2)


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    """Agreement beyond chance between two equal-length label lists."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def judge_arms(scores: dict[str, dict], ref: str, ask, cache: dict) -> dict:
    """M2 for every arm vs the reference on shared prompts."""
    from bench import judge as J
    out = {}
    for arm, arm_scores in scores.items():
        if arm == ref:
            continue
        w = t = l = flips = errors = 0
        per_prompt = {}
        for pid in sorted(set(arm_scores) & set(scores[ref])):
            v = J.compare(arm_scores[pid]["task"], arm_scores[pid]["text"], scores[ref][pid]["text"], ask, cache=cache)
            per_prompt[pid] = {k: v.get(k) for k in ("winner", "x_wins", "y_wins", "ties", "order_flips", "errored", "criteria")}
            errors += int(bool(v.get("errored")))
            flips += int(v.get("order_flips", 0))
            if v.get("errored"):
                continue
            w += v["winner"] == "x"
            l += v["winner"] == "y"
            t += v["winner"] == "tie"
        out[arm] = {"wins": w, "ties": t, "losses": l, "order_flips": flips, "errors": errors,
                    "sign_test_p": sign_test_p(w, l), "per_prompt": per_prompt}
    return out


def sign_test_p(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test on wins vs losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def kappa_vs_labels(labels_path: Path, judge_cache: dict) -> dict:
    """Pool every criterion of every human-labelled pair that the judge also
    judged. Labels and judge verdicts are both x / y / tie on the same pair key."""
    from bench.judge import CRITERIA
    human, machine = [], []
    for line in Path(labels_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        verdict = judge_cache.get(row.get("pair_key"))
        if not verdict or verdict.get("errored"):
            continue
        for c in CRITERIA:
            if c in (row.get("criteria") or {}) and c in (verdict.get("criteria") or {}):
                human.append(row["criteria"][c])
                machine.append(verdict["criteria"][c])
    k = cohens_kappa(human, machine)
    return {"n_judgements": len(human), "kappa": None if k is None else round(k, 3),
            "raw_agreement": round(sum(1 for a, b in zip(human, machine) if a == b) / len(human), 3) if human else None}


def build(run_dirs: list[Path], slate, *, ref: str = "B1", citations: str = "offline",
          ask=None, judge_cache: dict | None = None, labels: Path | None = None, check=check_report) -> dict:
    runs = [load_run(Path(d)) for d in run_dirs]
    arms: dict[str, dict] = {}
    skipped: dict[str, dict] = {}
    for run in runs:
        scored = score_run(run, slate, citations=citations, check=check)
        arms.setdefault(run["arm"], {}).update(scored)          # several runs may form one arm
        skipped.setdefault(run["arm"], {}).update(run["skipped"])
    if ref not in arms:
        raise ValueError(f"reference arm {ref!r} is not among the runs ({', '.join(sorted(arms))})")
    all_prompts = sorted(set().union(*[set(s) for s in arms.values()]))
    result: dict = {"ref": ref, "citations_mode": citations, "arms": {}, "paired": {}, "judge": None, "kappa": None,
                    "missing": {a: sorted(set(all_prompts) - set(s)) for a, s in arms.items()},
                    "skipped": skipped, "runs": {r["arm"]: str(r["run_dir"].name) for r in runs}}
    for arm, s in arms.items():
        rows = list(s.values())
        block = {m: _mean_ci([r.get(m) for r in rows]) for m in METRICS}
        block["n_prompts"] = len(rows)
        for key in ("length_chars", "generation_output_tokens", "calls", "elapsed_s", "cost_usd"):
            vals = [r.get(key) or 0 for r in rows]
            block[key] = round(statistics.mean(vals), 1) if vals else 0
        block["cost_incomplete"] = any(r.get("cost_incomplete") for r in rows)
        for key in ("n_cited", "resolved", "not_found", "title_mismatch", "unverifiable"):
            vals = [r.get(key) for r in rows if isinstance(r.get(key), int)]
            block[key] = sum(vals) if vals else None
        betas = [r["beta"] for r in rows if isinstance(r.get("beta"), (int, float))]
        block["beta_mean"] = round(statistics.mean(betas), 3) if betas else None
        result["arms"][arm] = block
        if arm != ref:
            result["paired"][arm] = {m: paired(s, arms[ref], m) for m in METRICS}
    ns = [v["composite_v1"]["n"] for v in result["paired"].values() if "composite_v1" in v and v["composite_v1"].get("n")]
    result["min_detectable_dz"] = min_detectable_dz(min(ns)) if ns else None
    if ask is not None:
        cache = judge_cache if judge_cache is not None else {}
        result["judge"] = judge_arms(arms, ref, ask, cache)
        if labels:
            result["kappa"] = kappa_vs_labels(labels, cache)
    return result


def _fmt(block: dict | None, pct: bool = False) -> str:
    if not block or block.get("mean") is None:
        return "n/a"
    lo, hi = block["ci95"]
    return f"{block['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def render(result: dict) -> str:
    ref = result["ref"]
    order = sorted(result["arms"], key=lambda a: (a != ref, a))
    L = [f"# Arm comparison (reference: {ref})", "",
         f"Runs: " + ", ".join(f"{a} = `{r}`" for a, r in sorted(result["runs"].items())),
         f"Citation check: {result['citations_mode']}. Every arm re-scored from its report text by this tool.", ""]
    L += ["## Scores (mean [bootstrap 95% CI])", "",
          "| Arm | n | composite v2 | citations resolved | coverage | composite v1 (format-only citations) |",
          "|---|---|---|---|---|---|"]
    for a in order:
        b = result["arms"][a]
        L.append(f"| {a} | {b['n_prompts']} | {_fmt(b['composite_v2'])} | {_fmt(b['citations_ratio'])} | "
                 f"{_fmt(b['coverage'])} | {_fmt(b['composite_v1'])} |")
    L += ["", "## Citations (totals over the arm's reports)", "",
          "| Arm | cited | resolved | not found | title mismatch | unverifiable |", "|---|---|---|---|---|---|"]
    for a in order:
        b = result["arms"][a]
        L.append("| " + " | ".join([a] + [("n/a" if b[k] is None else str(b[k]))
                                          for k in ("n_cited", "resolved", "not_found", "title_mismatch", "unverifiable")]) + " |")
    L += ["", "## Cost (mean per prompt)", "",
          "| Arm | calls | generation output tokens | wall s | reported $ | report chars | beta (coverage co-failure) |",
          "|---|---|---|---|---|---|---|"]
    for a in order:
        b = result["arms"][a]
        cost = f"{'>= ' if b['cost_incomplete'] else ''}{b['cost_usd']:.2f}"
        L.append(f"| {a} | {b['calls'] or 'n/r'} | {b['generation_output_tokens']:.0f} | {b['elapsed_s']:.0f} | {cost} | "
                 f"{b['length_chars']:.0f} | {'n/a' if b['beta_mean'] is None else b['beta_mean']} |")
    L += ["", f"## Paired differences vs {ref} (arm minus {ref}, shared prompts only)", ""]
    for a in order:
        if a == ref:
            continue
        L += [f"### {a}", "", "| Metric | n | mean delta | 95% CI | t | p | d_z | CI excludes 0 |", "|---|---|---|---|---|---|---|---|"]
        for m in METRICS:
            p = result["paired"][a][m]
            if p.get("n", 0) < 2:
                L.append(f"| {m} | {p.get('n', 0)} | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            L.append(f"| {m} | {p['n']} | {p['mean_delta']:+.4f} | [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | "
                     f"{p['t']} | {p['p']} | {p['d_z']} | {'yes' if p['ci_excludes_zero'] else 'no'} |")
        L.append("")
    mdz = result.get("min_detectable_dz")
    L.append(f"Smallest paired effect this n could detect (alpha 0.05, power 0.8): d_z ~ {mdz if mdz is not None else 'n/a'}. "
             "A CI that includes zero means 'not detected at this n', not 'no difference'.")
    if result.get("judge"):
        L += ["", f"## Blind pairwise judge vs {ref} (both orders; an order flip is a tie)", "",
              "| Arm | wins | ties | losses | sign-test p | order flips | judge errors |", "|---|---|---|---|---|---|---|"]
        for a in order:
            if a == ref:
                continue
            j = result["judge"][a]
            L.append(f"| {a} | {j['wins']} | {j['ties']} | {j['losses']} | {j['sign_test_p']} | {j['order_flips']} | {j['errors']} |")
    if result.get("kappa"):
        k = result["kappa"]
        L += ["", f"## Judge vs human labels", "",
              f"Cohen's kappa = {k['kappa']} over {k['n_judgements']} criterion-level judgements "
              f"(raw agreement {k['raw_agreement']}). Below ~0.6 the judged table above should not be trusted."]
    miss = {a: m for a, m in result["missing"].items() if m}
    if miss or any(result["skipped"].values()):
        L += ["", "## Not scored", ""]
        for a in order:
            if result["missing"].get(a):
                L.append(f"- {a} has no report for: {', '.join(result['missing'][a])}")
            for pid, why in sorted((result["skipped"].get(a) or {}).items()):
                L.append(f"- {a} {pid}: {why}")
    return "\n".join(L) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare bench arms on shared prompts")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", default="B1")
    ap.add_argument("--slate", default=str(Path(__file__).parent / "slate_v1.toml"))
    ap.add_argument("--citations", default="offline", choices=["off", "offline", "live"])
    ap.add_argument("--judge", action="store_true", help="run the blind pairwise judge (live model calls)")
    ap.add_argument("--allow-same-family", action="store_true")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=None, help="output dir (default: bench/comparisons/<ref>_<n runs>)")
    args = ap.parse_args()

    slate = load_slate(args.slate)
    out = Path(args.out) if args.out else Path(__file__).parent / "comparisons" / f"vs_{args.ref}"
    out.mkdir(parents=True, exist_ok=True)
    ask, cache = None, None
    if args.judge or args.labels:
        from bench import judge as J
        cache = J.load_cache(out / "judge_cache.json")
    if args.judge:
        from backends import make_role_router
        from config import load_config
        cfg = load_config()
        J.assert_cross_family(cfg, allow_same=args.allow_same_family)
        ask = J.make_ask(make_role_router(None, cfg), cfg, scratch_dir=out / "scratch")
    elif args.labels:
        ask = lambda p, s, schema: None          # cache-only: judge what is already judged
    result = build([Path(r) for r in args.runs], slate, ref=args.ref, citations=args.citations,
                   ask=ask, judge_cache=cache, labels=Path(args.labels) if args.labels else None)
    if cache is not None:
        J.save_cache(out / "judge_cache.json", cache)
    (out / "comparison.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (out / "comparison.md").write_text(render(result), encoding="utf-8")
    print(render(result))
    print(f"written: {out}/comparison.md")


if __name__ == "__main__":
    main()
