"""Prompt loader — domain-aware preset registry for role prompts.

Loads role prompts from `prompts/<domain>/<role>[_<persona>].md` with
fallback to `prompts/general/`. Caches loaded prompts and their SHAs
so the metrics layer can record exactly which prompt version was used
for any given run.

Used by `agents.py::plan_agents` to attach per-agent system prompts.

Layout:
    prompts/
        general/
            researcher.md
            critic.md
            critic_factual.md       # MAR persona (Phase 1 mechanism)
            critic_logical.md
            critic_counterfactual.md
            critic_steelman.md
            fact_checker.md
            quality_judge.md
            synthesizer.md
        crypto/
            researcher.md
            critic.md
            ...
        <future-domain>/
            ...
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPTS_ROOT = Path(__file__).parent
DEFAULT_DOMAIN = "general"
FALLBACK_DOMAIN = "general"


@dataclass(slots=True, frozen=True)
class PromptSet:
    """One loaded role prompt."""

    role: str
    persona: str | None
    domain: str
    template_path: Path
    system: str
    sha256: str


class PromptLoaderError(Exception):
    """Raised when no prompt file can be located for a given (role, persona)."""


class PromptLoader:
    """Domain-aware role-prompt loader with fallback to general/."""

    def __init__(
        self,
        *,
        root: Path = DEFAULT_PROMPTS_ROOT,
        domain: str = DEFAULT_DOMAIN,
        fallback_domain: str = FALLBACK_DOMAIN,
    ):
        self.root = root
        self.domain = domain
        self.fallback_domain = fallback_domain
        self._cache: dict[tuple[str, str | None, str], PromptSet] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, role: str, *, persona: str | None = None) -> PromptSet:
        """Load (role, persona) for the configured domain.

        Tries: prompts/<domain>/<role>[_<persona>].md
        Falls back to: prompts/<fallback_domain>/<role>[_<persona>].md
        Raises PromptLoaderError if neither exists.
        """
        cache_key = (role, persona, self.domain)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._resolve(role, persona, self.domain)
        if path is None and self.domain != self.fallback_domain:
            path = self._resolve(role, persona, self.fallback_domain)

        if path is None:
            raise PromptLoaderError(
                f"No prompt file found for role={role!r} persona={persona!r} "
                f"domain={self.domain!r} (also tried {self.fallback_domain!r})"
            )

        return self._load_path(role, persona, path)

    def list_personas(self, role: str) -> list[str]:
        """Return persona names for which a prompt file exists in the active domain.

        Looks for files matching `<role>_<persona>.md` and returns the persona
        suffixes. Searches both the active domain and the fallback.
        """
        domains = [self.domain]
        if self.fallback_domain != self.domain:
            domains.append(self.fallback_domain)

        seen: set[str] = set()
        out: list[str] = []
        for d in domains:
            domain_dir = self.root / d
            if not domain_dir.exists():
                continue
            prefix = f"{role}_"
            for p in sorted(domain_dir.iterdir()):
                if not p.is_file() or p.suffix != ".md":
                    continue
                if not p.name.startswith(prefix):
                    continue
                persona = p.stem[len(prefix):]
                if persona and persona not in seen:
                    seen.add(persona)
                    out.append(persona)
        return out

    def reload(self) -> None:
        """Drop the cache. Useful after editing prompt files in-place."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, role: str, persona: str | None, domain: str) -> Path | None:
        """Return the path to the prompt file, or None if missing."""
        filename = f"{role}_{persona}.md" if persona else f"{role}.md"
        path = self.root / domain / filename
        return path if path.is_file() else None

    def _load_path(self, role: str, persona: str | None, path: Path) -> PromptSet:
        text = path.read_text(encoding="utf-8")
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Domain may have come from fallback — recover it from the path
        domain_used = path.parent.name
        prompt_set = PromptSet(
            role=role,
            persona=persona,
            domain=domain_used,
            template_path=path,
            system=text.strip(),
            sha256=sha,
        )
        self._cache[(role, persona, self.domain)] = prompt_set
        return prompt_set


# ----------------------------------------------------------------------
# Persona assignment policy (MAR — Multi-Agent Reflexion, arXiv 2512.20845)
# ----------------------------------------------------------------------

PERSONA_ASSIGNMENT: dict[int, list[str]] = {
    1: ["factual"],
    2: ["factual", "counterfactual"],
    3: ["factual", "logical", "counterfactual"],
    # 4+ slots: rotate steelman in for round 2+ when dissent exists; cap at 3 unique personas otherwise
}


def assign_personas(
    critic_count: int,
    *,
    round_n: int = 1,
    has_unresolved_dissent: bool = False,
) -> list[str]:
    """Decide which persona each critic slot gets.

    - 1 slot: factual
    - 2 slots: factual + counterfactual
    - 3 slots: factual + logical + counterfactual
    - Round 2+ with unresolved dissent: replace last slot with steelman
    """
    base_count = max(1, min(critic_count, 3))
    base = list(PERSONA_ASSIGNMENT[base_count])
    if round_n >= 2 and has_unresolved_dissent and len(base) >= 1:
        base[-1] = "steelman"
    # If caller asked for more critics than we have unique personas, recycle.
    while len(base) < critic_count:
        base.append(base[len(base) % len(PERSONA_ASSIGNMENT[base_count])])
    return base
