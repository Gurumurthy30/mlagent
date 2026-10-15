# mlagent

A multi-agent command line system that runs a machine learning project from start to finish.
You go to the project folder, run one command, say what the task is, and the agents explore the
data, plan the validation, build features, train and check models, tune the best ones, and
write a report and a predictions file. All language model calls go to a local
[Ollama](https://ollama.com) model through `langchain-ollama`. The agents are wired together
with LangGraph.

It is meant for any kind of ML task: tabular, images, text, audio, time series, supervised or
not. The agents are told what the task is in your own words, and they write the code that fits.

## Requirements

- Python 3.11 or newer (Python 3.12 tested and recommended)
- Ollama running locally (or remote endpoint), with the model from `config.yaml` already pulled (default `gemma4:31b`)
- A machine that can run the code the agents write. Image, text and audio tasks benefit from a GPU.

## Install

### Windows (PowerShell)

If PowerShell blocks script execution (`UnauthorizedAccess`), set the process execution policy first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Windows (Command Prompt - cmd.exe)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Extra Dependencies

Extra libraries for deep learning, audio, vision, and development:

```powershell
pip install -e ".[deep]"        # torch, transformers, timm, sentence-transformers
pip install -e ".[audio]"       # librosa, soundfile
pip install -e ".[vision]"      # opencv-python
pip install -e ".[dev]"         # pytest, ruff, black
```

### GPU Support on Windows (PyTorch CUDA)

The default `torch` wheel on PyPI is CPU-only on Windows. For NVIDIA GPU acceleration, install the CUDA build first before running `pip install -e ".[deep]"`:

```powershell
# Example for CUDA 12.4 (see https://pytorch.org/get-started/locally for other versions)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify your Windows environment and GPU detection at any time:

```powershell
python -m mlagent.environment_probe
```

If it says `PyTorch: ... CPU only` while an NVIDIA GPU is available, the CUDA build is missing.

### Ollama on Windows

1. Download and install [Ollama for Windows](https://ollama.com/download/windows). Ollama runs as a background service at `http://localhost:11434`.
2. Open PowerShell and pull the model configured in `config.yaml`:
   ```powershell
   ollama pull gemma4:31b
   ```
3. Test connectivity in PowerShell:
   ```powershell
   Invoke-RestMethod http://localhost:11434/api/tags
   ```

The agents may also run `pip install` automatically in your environment when a library is missing. Every command they run is written to `mlagent_workspace/logs/commands.log`.

## Use

```powershell
cd C:\path\to\your\ml\project
mlagent start
```

*Note on Windows:* If the `mlagent` executable is not in your current shell `PATH`, invoke the module directly:

```powershell
python -m mlagent.cli start
# or:
python -m mlagent start
```

It asks two things, in your own words:

1. **The task**, for example: `"Predict which customers cancel next month. Train is train.csv, test is test.csv."`
2. **The metrics**, for example: `"macro F1 mainly, also report AUC"`. You may list several, or press Enter and let the agents decide.

After that it runs autonomously. It prints one line per step. When it finishes it shows where the report and the predictions are.

- **Ctrl+C** stops safely. Run `mlagent start` again and answer `resume`. An interrupted step is resumed cleanly from its beginning.
- If a workspace already exists, it asks `resume` or `fresh`. A fresh start renames the old workspace to `mlagent_workspace_old_<time>`. Nothing is deleted, and your edited `config.yaml` is carried over.
- `mlagent --version` displays version information.
- `mlagent init-config` generates a starter `config.yaml` with basic keys uncommented and advanced keys documented.
- `mlagent config` prints the effective configuration and source for every key.
- Override settings from the CLI with `--config <path>` or repeatable `--set section.key=value`.

## What it creates

Everything goes into one folder inside your project. Your own files are never changed.

```
mlagent_workspace/
  config.yaml     the only settings you edit
  memory/         ledger.json (all state), environment.json, progress.md (readable summary)
  logs/           run.log, llm_calls.jsonl (every prompt and answer), commands.log
  artifacts/      splits.json, features/, experiment_001/ ..., evaluation/, reviewer/
  report/         report.md and the final predictions file
```

## Configuration

`config.yaml` can be created manually, generated with `mlagent init-config`, or automatically created on the first run.
Configuration is split into two tiers:
- **BASIC**: ~15 commonly tuned settings (uncommented by default in `config.yaml`).
- **ADVANCED**: Finely tuned limits, context budgets, and evaluation thresholds (commented with descriptions in `config.yaml`).

Precedence is: **CLI flag (`--set`) > Config file (`config.yaml`) > Built-in defaults**.

### Configuration Reference

| Section | Key | Default | Tier | Description |
|---|---|---|---|---|
| `llm` | `name` | `gemma4:31b` | basic | Ollama model name (e.g. gemma4:31b, qwen2.5:32b) |
| `llm` | `base_url` | `https://ollama.com` | basic | Ollama server host address |
| `llm` | `temperature` | `0.2` | advanced | Sampling temperature for the LLM |
| `llm` | `context_length` | `16384` | basic | Ollama context window in tokens (num_ctx) |
| `llm` | `request_timeout_seconds` | `900` | advanced | HTTP request timeout for LLM calls in seconds |
| `llm` | `retries` | `2` | advanced | Maximum retries for JSON schema repair attempts |
| `compute` | `use_gpu` | `auto` | basic | GPU utilization mode: auto (ML Lead decides per task), always, or never |
| `limits` | `max_experiment_loops` | `15` | basic | Maximum rounds of the experiment loop |
| `limits` | `max_eda_loops` | `4` | basic | Maximum rounds for Data Analyst during EDA |
| `limits` | `max_data_engineer_rounds` | `3` | advanced | Maximum rounds for Data Engineer inspection |
| `limits` | `max_data_reentries` | `2` | advanced | Maximum times EDA can be reopened on Reviewer request |
| `limits` | `code_timeout_minutes` | `30` | basic | Execution time limit for a single script run in minutes |
| `limits` | `max_code_attempts` | `4` | basic | Maximum retry attempts for Coder to fix script errors |
| `limits` | `max_consecutive_crashes` | `3` | advanced | Stop rule threshold for consecutive script crashes |
| `limits` | `plateau_rounds` | `3` | advanced | Stop rule patience: consecutive rounds with no counted gain |
| `limits` | `min_relative_gain` | `0.001` | advanced | Minimum relative margin for counted gain when splits are few |
| `limits` | `tune_trial_count` | `20` | advanced | Number of Optuna tuning trials per tuning round |
| `limits` | `candidates_per_feature_round` | `5` | advanced | Maximum candidate feature ideas proposed per round |
| `evaluation` | `score_tolerance_relative` | `0.02` | advanced | Relative mismatch allowed between logged and recomputed validation scores |
| `evaluation` | `holdout_gap_std_multiplier` | `3.0` | advanced | Multiplier of CV standard deviation allowed for holdout score drop |
| `evaluation` | `holdout_gap_relative_limit` | `0.05` | advanced | Maximum relative drop allowed on holdout score before flagging |
| `evaluation` | `large_score_jump_std_multiplier` | `3.0` | advanced | Score jump threshold (std multiplier) that triggers rejection if leakage flag is set |
| `context` | `data_engineer` | `8500` | advanced | Digest character budget for Data Engineer |
| `context` | `data_analyst` | `5000` | advanced | Digest character budget for Data Analyst |
| `context` | `ml_lead` | `4500` | advanced | Digest character budget for ML Lead |
| `context` | `feature_engineer` | `16000` | advanced | Digest character budget for Feature Engineer |
| `context` | `modeler` | `7000` | advanced | Digest character budget for Modeler |
| `context` | `evaluator` | `5500` | advanced | Digest character budget for Evaluator |
| `context` | `tuner` | `5500` | advanced | Digest character budget for Tuner |
| `context` | `reviewer` | `5500` | advanced | Digest character budget for Reviewer |
| `context` | `submitter` | `4000` | advanced | Digest character budget for Submitter |
| `context` | `coder` | `8000` | advanced | Digest character budget for Coder |
| `context` | `report_facts` | `14000` | advanced | Context character budget for final report facts |
| `workspace` | `workspace_folder_name` | `mlagent_workspace` | basic | Folder name inside the project directory used for artifacts and memory |
| `workspace` | `archived_workspace_prefix` | `mlagent_workspace_old_` | advanced | Prefix for archived previous workspace runs |
| `workspace` | `big_file_mb` | `200` | advanced | File size in megabytes above which files are flagged as large |
| `workspace` | `max_files_in_listing` | `40` | advanced | Maximum number of files shown in data engineer directory listing |
| `workspace` | `ignored_folders` | `['.git', '.venv', 'venv', '__pycache__', 'node_modules']` | advanced | Folder names to skip when scanning the project |
| `machine` | `low_ram_warning_gb` | `4.0` | advanced | Free system RAM in GB below which a low memory warning is shown |
| `report` | `max_words` | `900` | basic | Target word count limit for the final Markdown report |
| `ui` | `mode` | `auto` | basic | Terminal display mode: auto, rich, or plain |
| `ui` | `theme` | `dark` | basic | Color theme: dark, light, or mono |
| `ui` | `verbosity` | `normal` | basic | Event logging verbosity: quiet, normal, or detailed |
| `ui` | `timestamps` | `True` | advanced | Show timestamps on log output lines |
| `ui` | `show_environment` | `True` | advanced | Display environment probe check table on startup |
| `ui` | `confirm_before_start` | `True` | advanced | Ask confirmation before starting execution |
| `ui` | `ascii_only` | `False` | advanced | Use ASCII characters only instead of Unicode symbols |
| `ui` | `log_width` | `100` | advanced | Terminal output line width limit |


## The team

Ten agents. Every agent that needs code hands a task to the **Coder**, which writes the script,
runs it, reads the error and fixes it.

| Agent | Job |
|---|---|
| Data Engineer | Opens the folder, reads your words, writes the project brief |
| Data Analyst | Exploratory data analysis with open questions and findings (self-loop) |
| ML Lead | Validation plan, metric, baseline, `splits.json` |
| Feature Engineer | Input preparation, screens candidate ideas on fixed splits |
| Modeler | Trains models, rejects repeated setups |
| Evaluator | Independent check of every result, hard rules and gain checks |
| Tuner | Bounded parameter search of best models (no LLM) |
| Reviewer | Evaluates rounds, tracks lessons, guides focus, and writes final report |
| Submitter | Mechanical delivery checks and optional ensemble |
| Coder | The sub-agent that writes and runs code for the others |

```mermaid
flowchart TD
  A([mlagent start]) --> DE[Data Engineer] --> DA[Data Analyst]
  DA -->|loop until eda.done| DA
  DA -->|done| ML[ML Lead] --> SC{stop check}
  SC -->|start or features| FE[Feature Engineer]
  SC -->|model| M[Modeler]
  SC -->|tune| T[Tuner]
  SC -->|data| DR[Data Analyst revisit]
  FE --> M
  DR --> FE
  M --> E[Evaluator]
  T --> E
  E --> R[Reviewer] --> SC
  SC -->|stop| S[Submitter] --> RP[Report] --> Z([report and predictions])
```

The loop stops for one of these reasons, decided by code and not by the language model: the
Reviewer finished (only allowed after enough rounds), the target score was reached, three runs
crashed in a row, the round limit was reached, three rounds in a row had no counted gain, or
the language model became unavailable. In every case the best approved result so far is
delivered.

A "counted gain" means a new result beats the best one by more than its noise: the mean of the
per-split improvements must exceed its standard error (three or more splits), or a small
relative margin otherwise.

## Testing

Install the development dependencies:

```powershell
pip install -e ".[dev]"
```

### Running Tests

Run the full test suite (skips live Ollama tests by default):

```powershell
pytest
```

Run with test coverage report:

```powershell
pytest --cov=mlagent --cov-report=term-missing
```

Run static lint checks:

```powershell
ruff check .
```

### Test Markers

Tests are organized with explicit pytest markers:
- `unit`: Fast invariant, property, and contract tests on state, config, digest, routing, prompts, and scoring.
  ```powershell
  pytest -m unit
  ```
- `integration`: End-to-end LangGraph pipeline tests using fake LLM and canned Coder outputs.
  ```powershell
  pytest -m integration
  ```
- `live`: Tests requiring an active local Ollama server (skipped by default unless `MLAGENT_LIVE=1`).
  ```powershell
  $env:MLAGENT_LIVE="1"; pytest -m live
  ```

### Hypothesis Profiles & Seed Reproduction

Property-based tests generate random ledgers and briefs using [Hypothesis](https://hypothesis.readthedocs.io/):
- `HYPOTHESIS_PROFILE=dev` (default): Fast local execution (50 examples per property).
- `HYPOTHESIS_PROFILE=ci`: CI matrix execution (100 examples per property).
- `HYPOTHESIS_PROFILE=thorough`: Scheduled deep test runs (500 examples per property).

Every test run outputs a reproducible session seed in the header:
```
mlagent session seed: 720004 (reproduce with --mlagent-seed=720004)
```

To reproduce any failure with the exact same pseudo-random sequence:
```powershell
pytest --mlagent-seed=720004
# or via environment variable:
$env:MLAGENT_TEST_SEED="720004"; pytest
```

## Honest limits

- It has been tested with a scripted language model only. How well your real Ollama model
  follows the prompts (one python block per answer, valid JSON) is only known from a live run.
  If a run misbehaves, `logs/llm_calls.jsonl` and `logs/run.log` show exactly what happened.
- There is no sandbox. The scripts run as you, with your permissions. The agents are told to
  keep your project folder read only and a small deny-list blocks obviously destructive shell
  commands (`rm -rf`, `shutdown` and similar), but that is a safety net and not isolation.
- The automated tests use a tabular project. The agents are written without per-data-type
  branches, but image, text and audio runs depend on your hardware and on the model writing
  good code for them.
- The time limit applies to one script run. The total time of a run is the sum of all of them.
