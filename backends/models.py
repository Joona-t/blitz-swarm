"""Role -> tier -> model resolution.

Code names roles ("planner", "synthesizer", ...) and abstract tiers
("fast", "standard", "deep"). Config maps roles to tiers (`[models]`) and
tiers to provider-native model strings (`[backend.providers.<id>.tiers]`).
This module is the single place where the two meet; no `.py` file holds a
literal model name.

Resolution order for `resolve_model(role, backend_id, cfg, override=None)`:

1. explicit `override` (CLI `--model`), else env `BLITZ_MODEL`
2. env `BLITZ_MODEL_<ROLE>` (role upper-cased, non-alphanumerics -> `_`)
3. `[models].<role>` -> tier -> `[backend.providers.<id>.tiers].<tier>`
4. `[models].default` -> tier -> provider tiers
5. provider `model` (legacy single-model key)
6. `ModelResolutionError` naming role, backend and tier. Never a silent
   literal default.

An override that names one of the provider's tiers ("deep") resolves
through the tier table; anything else is taken verbatim as a model string.
The returned `effort` is `[backend.providers.<id>.effort].<tier>` when set,
else the provider-wide `reasoning_effort`, else None.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

ENV_OVERRIDE = "BLITZ_MODEL"
ENV_ROLE_PREFIX = "BLITZ_MODEL_"


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    model: str
    tier: str
    effort: str | None
    backend_id: str
    source: str = ""   # override | env_role | tier | default_tier | legacy


class ModelResolutionError(LookupError):
    """No model could be resolved; the message names role, backend, tier."""

    def __init__(self, role: str, backend_id: str | None, tier: str | None, detail: str = ""):
        self.role = role
        self.backend_id = backend_id
        self.tier = tier
        msg = f"cannot resolve model for role={role!r} backend={backend_id!r} tier={tier!r}"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)


def role_env_var(role: str) -> str | None:
    """`BLITZ_MODEL_<ROLE>` for a role name; None when the role has no letters."""
    key = re.sub(r"[^A-Z0-9]+", "_", role.upper()).strip("_")
    return f"{ENV_ROLE_PREFIX}{key}" if key else None


def resolve_model(
    role: str,
    backend_id: str | None,
    cfg,
    override: str | None = None,
    *,
    env=None,
) -> ResolvedModel:
    """Resolve the model string for `role` on `backend_id` (see module doc)."""
    env = os.environ if env is None else env
    backend_id = backend_id or cfg.backend.default or None
    if not backend_id:
        raise ModelResolutionError(
            role, backend_id, None,
            "no backend selected: set [backend] default in blitz.toml or pass --backend",
        )
    try:
        provider = cfg.backend.get_provider(backend_id)
    except KeyError:
        raise ModelResolutionError(
            role, backend_id, None,
            f"backend {backend_id!r} is not defined under [backend.providers]",
        ) from None

    tiers: dict = provider.tiers or {}
    effort_table: dict = provider.effort or {}
    role_tier = cfg.models.roles.get(role) or None
    default_tier = cfg.models.default or None
    tier_hint = role_tier or default_tier

    def _done(model: str, tier: str, source: str) -> ResolvedModel:
        effort = effort_table.get(tier) or provider.reasoning_effort or None
        return ResolvedModel(model=model, tier=tier, effort=effort,
                             backend_id=backend_id, source=source)

    # 1. explicit override / BLITZ_MODEL
    explicit = override or env.get(ENV_OVERRIDE) or None
    if explicit:
        if explicit in tiers:
            return _done(tiers[explicit], explicit, "override")
        return _done(explicit, tier_hint or "override", "override")

    # 2. BLITZ_MODEL_<ROLE>
    var = role_env_var(role)
    role_env = env.get(var) if var else None
    if role_env:
        if role_env in tiers:
            return _done(tiers[role_env], role_env, "env_role")
        return _done(role_env, tier_hint or "override", "env_role")

    # 3. role tier
    if role_tier and role_tier in tiers:
        return _done(tiers[role_tier], role_tier, "tier")

    # 4. default tier
    if default_tier and default_tier in tiers:
        return _done(tiers[default_tier], default_tier, "default_tier")

    # 5. legacy single model
    if provider.model:
        return _done(provider.model, tier_hint or "legacy", "legacy")

    # 6. fail loudly
    have = ", ".join(sorted(tiers)) or "none"
    raise ModelResolutionError(
        role, backend_id, tier_hint,
        f"no entry for tier {tier_hint!r} in [backend.providers.{backend_id}.tiers] "
        f"(defined tiers: {have}) and no legacy 'model' key",
    )


def pricing_for(provider, model: str) -> dict | None:
    """`{input_per_mtok, output_per_mtok}` for `model` from the provider's
    `pricing` table, or None when unknown (callers report n/a, never 0)."""
    price = (provider.pricing or {}).get(model)
    if not isinstance(price, dict):
        return None
    return {
        "input_per_mtok": float(price["input_per_mtok"]),
        "output_per_mtok": float(price["output_per_mtok"]),
    }


def list_models(cfg, backend_id: str | None = None, override: str | None = None, *, env=None) -> list[dict]:
    """One row per configured role (+ a `*` row for any other role).

    Rows never raise: a role that cannot be resolved carries its error text,
    so `--list-models` stays dry-run friendly.
    """
    roles = sorted(cfg.models.roles) + ["*"]
    rows: list[dict] = []
    for role in roles:
        try:
            r = resolve_model(role, backend_id, cfg, override, env=env)
            rows.append({
                "role": role, "tier": r.tier, "model": r.model,
                "effort": r.effort or "-", "backend": r.backend_id,
                "source": r.source, "error": None,
            })
        except ModelResolutionError as exc:
            rows.append({
                "role": role, "tier": exc.tier or "-", "model": "-", "effort": "-",
                "backend": exc.backend_id or "-", "source": "error", "error": str(exc),
            })
    return rows


def format_models_table(cfg, backend_id: str | None = None, override: str | None = None, *, env=None) -> str:
    rows = list_models(cfg, backend_id, override, env=env)
    backend = rows[0]["backend"] if rows else (backend_id or cfg.backend.default)
    header = f"Resolved models for backend={backend!r} (default tier={cfg.models.default!r})"
    lines = [header, f"{'role':<16}{'tier':<10}{'model':<28}{'effort':<8}source"]
    for r in rows:
        lines.append(f"{r['role']:<16}{r['tier']:<10}{r['model']:<28}{r['effort']:<8}{r['source']}")
        if r["error"]:
            lines.append(f"    ! {r['error']}")
    lines.append(
        "Override: --model <model|tier>, env BLITZ_MODEL, env BLITZ_MODEL_<ROLE>; "
        "'*' = any role not listed under [models]."
    )
    return "\n".join(lines)


__all__ = [
    "ENV_OVERRIDE",
    "ENV_ROLE_PREFIX",
    "ModelResolutionError",
    "ResolvedModel",
    "format_models_table",
    "list_models",
    "pricing_for",
    "resolve_model",
    "role_env_var",
]
