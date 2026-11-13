"""Phase 1 WP8-9: cross-arm comparison, judge aggregation, human labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import compare as C
from bench import judge as J
from bench import label as L
from bench import load_slate

SLATE = Path(__file__).resolve().parents[2] / "bench" / "slate_v1.toml"
PIDS = ["s001", "s002", "s003", "s004", "s005"]


def _make_run(tmp_path, arm, texts: dict[str, str], *, errored=(), calls=1):
    run = tmp_path / f"run_{arm}"
    (run / "reports").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"arm": arm}), encoding="utf-8")
    rows = []
    for pid, text in texts.items():
        (run / "reports" / f"{pid}.md").write_text(text, encoding="utf-8")
        rows.append({"prompt_id": pid, "output_md_path": f"bench/runs/x/{pid}.md", "error": None, "timeout": False,
                     "output_tokens": 1000 * calls, "calls": calls, "elapsed_s": 10.0 * calls, "cost_usd": 0.0,
                     "cost_incomplete": True, "beta": {"beta": 0.2} if arm != "B1" else None})
    for pid in errored:
        rows.append({"prompt_id": pid, "error": "budget_exhausted", "timeout": False})
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return run


def _text(slate, pid, n_keywords, cite=""):
    kws = list(slate.by_id(pid).expected_coverage)[:n_keywords]
    return "# Report\n\n" + " ".join(kws) + "\n\n" + cite


def _fake_check(text, offline=False):
    real = text.count("arXiv:1706.03762")
    fake = text.count("arXiv:2301.99991")
    n = real + fake
    return {"n_cited": n, "resolved": real, "title_mismatch": 0, "not_found": fake, "unverifiable": 0,
            "ratio": (real / n) if n else None, "items": []}


def test_planted_gain_shows_with_right_sign_n_and_missing_prompts_listed(tmp_path):
    slate = load_slate(SLATE)
    # B1 covers 2 keywords everywhere; S covers 4 on four prompts and errored on s005.
    b1 = _make_run(tmp_path, "B1", {p: _text(slate, p, 2) for p in PIDS})
    s = _make_run(tmp_path, "S", {p: _text(slate, p, 4) for p in PIDS[:4]}, errored=("s005",), calls=20)
    result = C.build([s, b1], slate, ref="B1", citations="off")
    assert result["arms"]["S"]["n_prompts"] == 4 and result["arms"]["B1"]["n_prompts"] == 5
    assert result["missing"]["S"] == ["s005"] and result["skipped"]["S"] == {"s005": "budget_exhausted"}
    cov = result["paired"]["S"]["coverage"]
    assert cov["n"] == 4 and cov["mean_delta"] > 0.3 and cov["ci_excludes_zero"] is True
    assert result["paired"]["S"]["composite_v2"]["n"] == 0, "no citation check -> v2 is not invented"
    assert result["arms"]["S"]["calls"] == 20 and result["arms"]["S"]["beta_mean"] == 0.2
    md = C.render(result)
    assert "S has no report for: s005" in md and "S s005: budget_exhausted" in md
    assert "Smallest paired effect" in md and "| S | 4 |" in md


def test_composite_v2_separates_real_from_fabricated_citations(tmp_path):
    slate = load_slate(SLATE)
    real = "see arXiv:1706.03762 and arXiv:1706.03762v2 text arXiv:1706.03762"
    fake = "see arXiv:2301.99991"
    b1 = _make_run(tmp_path, "B1", {p: _text(slate, p, 3, "source arXiv:1706.03762") for p in PIDS})
    s = _make_run(tmp_path, "S", {p: _text(slate, p, 3, fake) for p in PIDS})
    result = C.build([s, b1], slate, ref="B1", citations="offline", check=_fake_check)
    assert result["arms"]["S"]["citations_ratio"]["mean"] == 0.0
    assert result["arms"]["B1"]["citations_ratio"]["mean"] == 1.0
    assert result["arms"]["S"]["not_found"] == 5 and result["arms"]["B1"]["resolved"] == 5
    d = result["paired"]["S"]["composite_v2"]
    assert d["n"] == 5 and d["mean_delta"] < 0, "fabricated citations must cost the arm on v2"
    assert abs(result["paired"]["S"]["composite_v1"]["mean_delta"]) < 1e-9, "v1 cannot tell them apart"


def test_missing_reference_arm_is_an_error(tmp_path):
    slate = load_slate(SLATE)
    s = _make_run(tmp_path, "S", {p: _text(slate, p, 2) for p in PIDS})
    with pytest.raises(ValueError, match="reference arm 'B1'"):
        C.build([s], slate, ref="B1", citations="off")


def test_stat_helpers():
    assert C.min_detectable_dz(20) == pytest.approx(0.66, abs=0.01)
    assert C.min_detectable_dz(34) == pytest.approx(0.49, abs=0.02)   # normal approximation of the power term; exact is 0.50
    assert C.min_detectable_dz(2) is None
    assert C.sign_test_p(5, 0) == pytest.approx(0.0625)
    assert C.sign_test_p(3, 3) == 1.0 and C.sign_test_p(0, 0) is None
    # hand-computed: po = 0.75, pe = 0.5*0.5 + 0.5*0.5 = 0.5 -> kappa 0.5
    assert C.cohens_kappa(["x", "x", "y", "y"], ["x", "x", "y", "x"]) == pytest.approx(0.5)
    assert C.cohens_kappa(["x", "y"], ["x", "y"]) == 1.0 and C.cohens_kappa([], []) is None


def test_judge_table_and_kappa_against_scripted_labels(tmp_path):
    slate = load_slate(SLATE)
    b1 = _make_run(tmp_path, "B1", {p: _text(slate, p, 2) + "plain" for p in PIDS})
    s = _make_run(tmp_path, "S", {p: _text(slate, p, 2) + "GOOD" for p in PIDS}, calls=20)

    def ask(prompt, system, schema):
        a = prompt.split("<<<REPORT A>>>")[1].split("<<<END REPORT A>>>")[0]
        return {**{c: ("A" if "GOOD" in a else "B") for c in J.CRITERIA}, "expected_facts": [], "rationale": ""}

    cache: dict = {}
    result = C.build([s, b1], slate, ref="B1", citations="off", ask=ask, judge_cache=cache)
    j = result["judge"]["S"]
    assert (j["wins"], j["ties"], j["losses"], j["order_flips"]) == (5, 0, 0, 0) and j["sign_test_p"] == 0.0625
    assert "| S | 5 | 0 | 0 |" in C.render(result)

    # Human labelling session: scripted stdin. The human agrees with the judge
    # on every criterion of every pair -> kappa is degenerate-perfect.
    arms = {r["arm"]: C.score_run(r, slate, citations="off") for r in (C.load_run(s), C.load_run(b1))}
    pairs = L.build_pairs(arms, "B1", n=3, seed=1)
    assert len(pairs) == 3 and all(p["pair_key"] in cache for p in pairs)
    answers = []
    for p in pairs:
        answers += ["a" if p["x_is_a"] else "b"] * len(J.CRITERIA)      # always pick the arm (GOOD) side
    it = iter(answers)
    labels = tmp_path / "labels" / "labels.jsonl"
    n = L.label_pairs(pairs, labels, input_fn=lambda q: next(it), show_fn=lambda a, b: None, out=lambda s: None)
    assert n == 3
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    assert all(set(r["criteria"].values()) == {"x"} for r in rows), "labels are stored un-randomised"
    k = C.kappa_vs_labels(labels, cache)
    assert k["n_judgements"] == 12 and k["raw_agreement"] == 1.0
    # resumable: a second sitting has nothing to do and asks nothing
    assert L.label_pairs(pairs, labels, input_fn=lambda q: (_ for _ in ()).throw(AssertionError("asked")),
                         show_fn=lambda a, b: None, out=lambda s: None) == 0


def test_label_quit_keeps_finished_pairs_only(tmp_path):
    pairs = [{"arm": "S", "prompt_id": f"p{i}", "task": "t", "x": f"x{i}", "y": f"y{i}",
              "pair_key": f"k{i}", "x_is_a": True} for i in range(2)]
    it = iter(["a", "b", "t", "a", "a", "q"])
    labels = tmp_path / "labels.jsonl"
    n = L.label_pairs(pairs, labels, input_fn=lambda q: next(it), show_fn=lambda a, b: None, out=lambda s: None)
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    assert n == 1 and len(rows) == 1
    assert rows[0]["criteria"] == {"accuracy": "x", "completeness": "y", "specificity": "tie", "consistency": "x"}


# ---- ITER-P1e: conditions ---------------------------------------------------


def _cond_run(tmp_path, arm, cond, texts, *, in_config=True):
    run = _make_run(tmp_path / cond, arm, texts)
    if in_config:
        (run / "config.json").write_text(json.dumps({"arm": arm, "condition": cond}), encoding="utf-8")
    return run


def test_two_runs_of_one_arm_on_the_same_prompts_are_refused_not_merged(tmp_path):
    slate = load_slate(SLATE)
    texts = {p: _text(slate, p, 2) for p in PIDS}
    a = _cond_run(tmp_path, "B1", "off", texts, in_config=False)
    b = _cond_run(tmp_path, "B1", "on", texts, in_config=False)
    with pytest.raises(ValueError, match="both have arm 'B1' for s001.*--condition"):
        C.build([a, b], slate, ref="B1", citations="off")
    # Disjoint prompts are one arm run in two parts and still merge.
    c = _cond_run(tmp_path, "B1", "part1", {p: _text(slate, p, 2) for p in PIDS[:2]}, in_config=False)
    d = _cond_run(tmp_path, "B1", "part2", {p: _text(slate, p, 2) for p in PIDS[2:]}, in_config=False)
    assert C.build([c, d], slate, ref="B1", citations="off")["arms"]["B1"]["n_prompts"] == 5


def test_each_arm_is_compared_with_the_reference_of_its_own_condition(tmp_path):
    slate = load_slate(SLATE)
    runs = [
        _cond_run(tmp_path, "B1", "off", {p: _text(slate, p, 2) for p in PIDS}),
        _cond_run(tmp_path, "S", "off", {p: _text(slate, p, 4) for p in PIDS}),     # S better with thinking off
        _cond_run(tmp_path, "B1", "on", {p: _text(slate, p, 4) for p in PIDS}),
        _cond_run(tmp_path, "S", "on", {p: _text(slate, p, 2) for p in PIDS}),      # and worse with it on
        _cond_run(tmp_path, "B3", "other", {p: _text(slate, p, 3) for p in PIDS}),  # no B1 in this condition
    ]
    result = C.build(runs, slate, ref="B1", citations="off")
    assert set(result["arms"]) == {"B1@off", "S@off", "B1@on", "S@on", "B3@other"}
    assert set(result["paired"]) == {"S@off", "S@on"}
    assert result["paired"]["S@off"]["coverage"]["mean_delta"] > 0.3
    assert result["paired"]["S@on"]["coverage"]["mean_delta"] < -0.3
    assert result["no_reference"] == ["B3@other"]
    md = C.render(result)
    assert "### S@off vs B1@off" in md and "### S@on vs B1@on" in md
    assert "B3@other has no B1@other run to be compared with" in md


def test_condition_can_be_given_for_runs_recorded_without_one(tmp_path):
    slate = load_slate(SLATE)
    texts = {p: _text(slate, p, 2) for p in PIDS}
    a = _cond_run(tmp_path, "B1", "off", texts, in_config=False)
    b = _cond_run(tmp_path, "B1", "on", texts, in_config=False)
    b = b.rename(b.parent / "run_B1_on")
    result = C.build([a, b], slate, ref="B1", citations="off", conditions={"run_B1_on": "on"})
    assert set(result["arms"]) == {"B1", "B1@on"}
