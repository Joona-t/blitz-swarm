# dependency-auditor Report: Blitz Swarm

_Agent: dependency-auditor | Reporting to: Linus (Security Lead) | Date: 2026-03-12_

## Summary
4 runtime deps declared. One (aiosqlite) is dead. Heaviest supply chain risk: sentence-transformers (pulls PyTorch ~2GB).

## Dependency Matrix
| Package | Declared | Imported | Actually Used | Risk |
|---------|----------|----------|---------------|------|
| redis>=5.0 | Yes | Yes (blackboard.py) | Yes | Low |
| aiosqlite>=0.20 | Yes | **NO** | **NO** | Dead dep — remove |
| lancedb>=0.6 | Yes | Yes (try/except) | Yes (optional) | Medium |
| sentence-transformers>=3.0 | Yes | Yes (lazy) | Yes | High (2-4GB transitive) |

## Key Findings
1. **aiosqlite never imported** — dead dependency, remove immediately
2. **sentence-transformers** pulls PyTorch, transformers, huggingface-hub — 2-4GB install
3. **Redis server CVE-2025-49844** (CVSS 10.0) — redis:7-alpine Docker tag should be pinned
4. **No version upper bounds** — lancedb has history of breaking minor-version changes
5. **No lockfile** — no requirements.lock or uv.lock
6. **redis and lancedb are functionally optional** (soft-imported) but declared as hard deps

## Recommendations
1. Remove aiosqlite
2. Make lancedb and sentence-transformers optional extras
3. Pin Redis Docker image to patched version
4. Add upper bounds to lancedb pin
5. Generate a lockfile
