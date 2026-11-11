"""Preflight for live bench runs (Phase 1, WP10).

Cheap, model-free checks that fail with a named reason BEFORE hours of
calls start: the CLIs exist, the judge is another family when the arm needs
one, the citation cache is writable, the disk has room, the budget source
exists. Returns a list of problems; empty = go. (A long job launched
without a preflight is a recorded failure class in the studio ledger.)
"""

from __future__ import annotations

import shutil
from pathlib import Path

MIN_FREE_GB = 2.0


def run_preflight(cfg, *, blitz_cfg=None, which=shutil.which, disk_usage=shutil.disk_usage) -> list[str]:
    problems: list[str] = []
    try:
        if blitz_cfg is None:
            from config import load_config
            blitz_cfg = load_config()
    except Exception as exc:
        return [f"blitz.toml does not load: {exc}"]

    from backends import make_provider_backend
    from backends.models import provider_for_role

    roles = ["researcher"]
    if cfg.arm in ("B2",):
        roles.append("judge")
    if cfg.arm == "S":
        roles += ["critic", "quality_judge", "synthesizer", "judge"]
    seen: set[str] = set()
    for role in roles:
        try:
            pid = provider_for_role(role, blitz_cfg, cfg.backend_id)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if not pid or pid in seen:
            continue
        seen.add(pid)
        try:
            backend = make_provider_backend(pid, blitz_cfg, with_fallback=False)
        except Exception as exc:
            problems.append(f"provider {pid!r} cannot be built: {exc}")
            continue
        inner = getattr(backend, "inner", backend)
        command = getattr(inner, "command", None)
        binary = command[0] if command else {"codex_cli": "codex", "claude_cli": "claude",
                                             "gemini_cli": "gemini"}.get(getattr(inner, "adapter_kind", ""))
        if binary and which(binary) is None:
            problems.append(f"provider {pid!r} needs `{binary}` on PATH (role {role})")

    if cfg.arm == "B2" and not cfg.allow_same_family:
        from bench.judge import SameFamilyJudge, assert_cross_family
        try:
            assert_cross_family(blitz_cfg, backend_id=cfg.backend_id)
        except SameFamilyJudge as exc:
            problems.append(str(exc))
        except ValueError as exc:
            problems.append(str(exc))

    if cfg.arm == "B1" and cfg.budget_from and not (Path(cfg.budget_from) / "results.jsonl").exists():
        problems.append(f"--budget-from {cfg.budget_from}: no results.jsonl there")
    if cfg.arm == "B1" and not cfg.budget_from and not cfg.output_token_budget:
        problems.append("arm B1 has no budget: pass --budget-from <S run dir> or --output-token-budget N "
                        "(without one it stops after 3 calls and is not compute-matched)")

    if cfg.citations in ("live", "offline"):
        from bench.citations import DEFAULT_CACHE
        try:
            DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            probe = DEFAULT_CACHE.parent / ".write_probe"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            problems.append(f"citation cache dir not writable: {exc}")

    try:
        free_gb = disk_usage(str(Path(__file__).parent)).free / 1e9
        if free_gb < MIN_FREE_GB:
            problems.append(f"only {free_gb:.1f} GB free; need {MIN_FREE_GB:.0f} GB")
    except OSError:
        pass
    return problems


__all__ = ["run_preflight", "MIN_FREE_GB"]
