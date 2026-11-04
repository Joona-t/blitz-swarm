# Agent rules for this repo

Rules an agent must follow here, each paired with what enforces it. A rule with
"docs only" is the weakest kind: nothing fails when it is skipped.

| Rule | Enforced by | Proven against |
|---|---|---|
| Never commit while the suite is red | `scripts/hooks/pre-commit` (pytest exit code), installed by `sh scripts/install-hooks.sh`; CI `tests.yml` | commit `1d4f38a` (red push, 2026-11-04); refused `test_zz_red_proof.py` |
| No model name literals outside `blitz.toml` | `tests/test_no_hardcoded_models.py` | WP5 first commit (ACP docstring), 2026-11-04 |
| No home paths in tracked files | `tests/test_packaging.py::test_no_home_paths_in_tracked_files`; CI `leak-gate` | `mythos-bench/results/runs.jsonl` (redacted), `jobs/blitz_upgrade.py` |
| No permission-bypass flags in production code | `tests/test_tool_policy.py::test_no_permission_bypass_flags_in_production_code` | `--dangerously-skip-permissions` on `main` (AUDIT H4) |
| Every adapter passes the conformance suite before it is called supported | `tests/test_conformance.py` (registry adapter missing from the table fails) | ACP adapter, 2026-11-04 |
| README harness table matches the adapters' declarations | `tests/test_docs_truth.py` | v0.4.0 |
| `--dry-run` and `--list-models` never call a model | `tests/test_provider_pins.py::test_list_models_cli_is_model_free` (PATH=/nonexistent) | PR #2 defect 5 |

Commit identity: `215574532+Joona-t@users.noreply.github.com`. Commits end with the
`Co-Authored-By` line for the model that made them.
