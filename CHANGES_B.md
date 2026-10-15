# Stage B: Tests & CI (`CHANGES_B.md`)

This document records the complete implementation of the Stage B test suite, GitHub Actions CI workflow, test inventory by module, bugs discovered and fixed during development, local execution instructions across platforms, and deviations.

---

## B.1 Executive Summary

Stage B established a robust, property- and invariant-based testing infrastructure designed to survive future refactorings and model changes. Testing asserts dynamic invariants, schema properties, contract adherence, process lifecycle control, and edge-case behavior rather than brittle prose, hardcoded loop counts, or exact model outputs.

Key accomplishments:
- **84 automated tests** across 9 test modules in `tests/`, covering unit invariants, property testing with Hypothesis, and end-to-end LangGraph pipeline integration.
- **Reproducible test sessions** using a custom pytest seed hook (`--mlagent-seed=<int>` and `MLAGENT_TEST_SEED`), ensuring any property-based failure can be reproduced deterministically.
- **Zero test failures** and 100% pass rate in offline environments.
- **Lint and type safety**: Full pass with `ruff check .` (0 errors) and `python -m pyflakes mlagent` (0 errors).
- **Cross-platform CI**: `.github/workflows/ci.yml` matrix across Ubuntu and Windows on Python 3.11 and 3.12, with dev, CI, and weekly thorough Hypothesis profiles.

---

## B.2 Test Inventory by Module

| Test Module | Tests | Markers | Focus & Invariants Tested |
|---|---|---|---|
| [`tests/test_state_memory.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_state_memory.py) | 8 | `unit` | Ledger JSON serialization/deserialization round-trip with Hypothesis; initial ledger invariants; save/load round-trip; `best_approved_run` metric directionality; query invariants; atomic crash safety fallback to previous backup; missing ledger load errors; human-readable progress file generation. |
| [`tests/test_config.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_config.py) | 7 | `unit` | Config defaults validation; field description completeness; numeric bounds and enum validation; unknown key rejection (`extra="forbid"`); `init-config` text round-trip; 3-tier precedence (`CLI flag > config.yaml > defaults`); Markdown table synchronization. |
| [`tests/test_digest.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_digest.py) | 5 | `unit` | Digest character budget caps across all 10 agents; elimination of literal `"None"` strings; whole-word boundary truncation; Reviewer EDA summary inclusion; Report facts sections completeness and formatting. |
| [`tests/test_routers_stop_rules.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_routers_stop_rules.py) | 7 | `unit` | Router targets valid within compiled graph; budget stop rule; consecutive crashes stop rule; plateau patience stop rule; target score stop rule; failure diversion to submit; Reviewer finish guard against premature termination before minimum rounds. |
| [`tests/test_scoring_evaluator.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_scoring_evaluator.py) | 7 | `unit` | Monotonicity of counted gain; noise guard on per-split variance; train/validation overlap rejection; validation score discrepancy rejection; holdout score drop rejection; large jump with leakage rejection; clean run approval. |
| [`tests/test_code_runner.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_code_runner.py) | 7 | `unit` | `<TAG>JSON</TAG>` extraction amidst noise; markdown code fence stripping inside tags; rejection of missing/malformed tags; dangerous shell command detection; python script execution; crash error capture; process tree killing on timeout; `mlagent_tools.py` loading. |
| [`tests/test_prompts.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_prompts.py) | 25 | `unit` | Placeholder-to-format parity for all 12 prompt templates across all agents; literal JSON brace escaping (`{{` and `}}`); prompt formatting without unexpanded placeholders; context budget limits under Hypothesis random ledgers. |
| [`tests/test_events.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_events.py) | 15 | `unit` | Emitting every `EventType` to subscribers; Rich markup safety (rendering `[red]`, `[bold]`, `[/]`, `[[nested]]` without exceptions or text loss); subscriber error isolation (listener crash does not halt emit). |
| [`tests/test_pipeline.py`](file:///c:/Users/gurum/Documents/4_Projects/ML_agent_project/tests/test_pipeline.py) | 3 | `integration` | Full graph execution using `FakeLlmClient` and canned Coder; execution of every agent at least once (Data Engineer, Data Analyst, ML Lead, Feature Engineer, Modeler, Evaluator, Reviewer, Tuner, Submitter); unique non-empty IDs for Finding and FeatureRecord; report generation with required sections; mid-run interruption and resume without duplicating finished phases; failure recovery diverting approved work to submit. |
| **Total** | **84** | | **All 84 Passed** |

---

## B.3 How to Run Locally

### 1. Installation
Install the package with `dev` dependencies:
```powershell
# Windows (PowerShell)
pip install -e ".[dev]"
```
```bash
# Linux / macOS
pip install -e ".[dev]"
```

### 2. Running Tests
Run the entire test suite (skipping live Ollama calls):
```powershell
pytest
```

Run with line-by-line missing coverage report:
```powershell
pytest --cov=mlagent --cov-report=term-missing
```

Run only unit tests:
```powershell
pytest -m unit
```

Run only integration pipeline tests:
```powershell
pytest -m integration
```

Run with live Ollama model (requires active Ollama server at localhost:11434):
```powershell
$env:MLAGENT_LIVE="1"
pytest -m live
```

### 3. Reproducing Failures with Seed
Every test run prints the session seed in the header:
```
mlagent session seed: 720004 (reproduce with --mlagent-seed=720004)
```

To reproduce any run identically:
```powershell
pytest --mlagent-seed=720004
# or using an environment variable:
$env:MLAGENT_TEST_SEED="720004"; pytest
```

### 4. Hypothesis Profiles
Adjust property-based test volume via `HYPOTHESIS_PROFILE`:
- `dev` (default): 50 examples per property.
- `ci`: 100 examples per property.
- `thorough`: 500 examples per property.

```powershell
$env:HYPOTHESIS_PROFILE="thorough"
pytest
```

### 5. Linter and Static Checks
```powershell
python -m ruff check .
python -m pyflakes mlagent
```

---

## B.4 Real Bugs Discovered & Fixed

During Stage B development and invariant testing, several subtle bugs were uncovered and resolved:

1. **`mlagent/digest.py:385` — Invalid f-string format specifier**:
   - *Problem*: In `build_report_facts`, f-string `{f.score_gain:+.4f if f.score_gain is not None else 'n/a'}` raised `ValueError: Invalid format specifier` because Python treated `if ... else` as format syntax.
   - *Fix*: Separated the formatting into `{f'{f.score_gain:+.4f}' if f.score_gain is not None else 'n/a'}`.

2. **`mlagent/code_runner.py:65` — Markdown fences inside `<TAG>JSON</TAG>`**:
   - *Problem*: When scripts or LLMs wrapped JSON inside `<RESULT>```json\n...\n```</RESULT>`, `extract_tagged_json` failed with `JSONDecodeError`.
   - *Fix*: Added automatic stripping of opening and closing markdown code fences before calling `json.loads`.

3. **`mlagent/state.py` & `mlagent/graph.py` — Phase literal mismatch**:
   - *Problem*: `graph.py` transitions to phase `"submit"` (`{"phase": "submit"}`), but `Phase` in `state.py` was typed as `Literal[..., "submitter", ...]`. This caused a Pydantic `ValidationError` inside LangGraph state coercion, abruptly failing the pipeline.
   - *Fix*: Added `"submit"` as an allowed alias in `Phase` literal in `state.py` and aligned graph transition conventions.

4. **`mlagent/graph.py:428` — Resuming failed/interrupted runs**:
   - *Problem*: When an interrupted run had `status == "failed"`, invoking `run_pipeline` immediately exited to `END` on the first node because `run_is_over` returned `True`.
   - *Fix*: In `run_pipeline`, if a ledger is being resumed with `status == "failed"` and `phase != "done"`, its status is automatically reset to `"running"` and `error_message` is cleared.

5. **`mlagent/agents/modeler.py:101,187` — Defensive scoping & formatting**:
   - *Problem*: `candidate` was only assigned inside a `range(3)` loop, and `run.score_mean:.4f` would raise `TypeError` if `run.score_mean` was `None` on crash-free zero-score edge cases.
   - *Fix*: Explicitly initialized `candidate: ModelChoice | None = None` before the loop and guarded `score_mean` formatting.

6. **`mlagent/agents/reviewer.py:199` — Ambiguous variable name**:
   - *Problem*: Single-letter variable `l` in comprehension flagged by Ruff rule `E741`.
   - *Fix*: Renamed to `line`.

---

## B.5 Anything Not Covered & Why

- **Live Ollama Inference in Default Test Runs**: Tests using live LLM connections are gated behind `@pytest.mark.live` and skipped by default. This ensures deterministic, fast, offline test execution without requiring an NVIDIA GPU or running Ollama daemon in developer and CI environments.
- **System-level Process Suspension / SIGKILL on Windows**: Process kill tests test `kill_process_tree` with `psutil`, which works identically on Windows and Linux, but kernel-level hardware faults cannot be simulated.

---

## B.6 Deviations

- **Git Branch**: The user explicitly instructed to remain on branch `simplify-state-agents` rather than creating a separate `stage-b-tests-ci` branch. All work was performed and validated directly on `simplify-state-agents`.
- **Flexible Invariant Design**: In accordance with user feedback, tests avoid brittle exact-match assertions on prose, log lines, or exact counts, focusing strictly on schemas, unique identifiers, monotonic gains, budget caps, and router invariants.
