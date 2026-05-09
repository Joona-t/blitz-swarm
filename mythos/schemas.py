"""Mythos Swarm — JSON schemas for structured agent output.

Passed to `claude -p --json-schema <schema>` to enforce shape.
Matches the schema discipline of agents.py::AGENT_OUTPUT_SCHEMA.
"""

import json

# ---------------------------------------------------------------------------
# Planner — decomposes a task into sub-specs + global invariants
# ---------------------------------------------------------------------------

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "task_summary": {
            "type": "string",
            "description": "One-paragraph restatement of the task in your own words.",
        },
        "sub_specs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Stable id like 'spec_00'"},
                    "title": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Concrete, executable description for the executor.",
                    },
                    "deliverable": {
                        "type": "string",
                        "description": "What artifact (file, function, proof step) the executor must produce.",
                    },
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Bullet list of conditions the deliverable must meet.",
                    },
                },
                "required": ["id", "title", "description", "deliverable", "acceptance_criteria"],
            },
            "description": "Sub-specs the executors will implement in parallel.",
        },
        "global_invariants": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Cross-cutting properties the assembled artifact must satisfy.",
        },
        "verification_strategy": {
            "type": "string",
            "description": "How the verifier should check correctness (test plan, proof obligations, equivalence checks).",
        },
        "rationale": {
            "type": "string",
            "description": "Why this decomposition? Why these invariants?",
        },
        "confidence": {
            "type": "number", "minimum": 0, "maximum": 1,
        },
    },
    "required": [
        "task_summary", "sub_specs", "global_invariants",
        "verification_strategy", "rationale", "confidence",
    ],
}

PLANNER_SCHEMA_JSON = json.dumps(PLANNER_SCHEMA)


# ---------------------------------------------------------------------------
# Executor — implements one sub-spec
# ---------------------------------------------------------------------------

EXECUTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "spec_id": {"type": "string"},
        "deliverable": {
            "type": "string",
            "description": "The actual produced artifact (code, prose, proof). Markdown OK.",
        },
        "acceptance_check": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "satisfied": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["criterion", "satisfied", "evidence"],
            },
            "description": "Self-check against each acceptance criterion.",
        },
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Things you suspect could fail verification — surface for the verifier.",
        },
        "confidence": {
            "type": "number", "minimum": 0, "maximum": 1,
        },
    },
    "required": ["spec_id", "deliverable", "acceptance_check", "confidence"],
}

EXECUTOR_SCHEMA_JSON = json.dumps(EXECUTOR_SCHEMA)


# ---------------------------------------------------------------------------
# Verifier — gate check across all executor outputs
# ---------------------------------------------------------------------------

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "needs_work"],
        },
        "invariant_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "invariant": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pass", "fail", "unverifiable"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["invariant", "status", "evidence"],
            },
        },
        "spec_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pass", "fail", "partial"],
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["spec_id", "status"],
            },
        },
        "required_fixes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete instructions the planner should use when replanning. Empty list when verdict=pass.",
        },
        "summary": {"type": "string"},
        "confidence": {
            "type": "number", "minimum": 0, "maximum": 1,
        },
    },
    "required": [
        "verdict", "invariant_results", "spec_results",
        "required_fixes", "summary", "confidence",
    ],
}

VERIFIER_SCHEMA_JSON = json.dumps(VERIFIER_SCHEMA)
