"""Fixed values used across the project.

Users only change config.yaml. The values here are the agents' working rules.
The ones that matter to an agent are also written into its prompt, so the agent knows its limits.
"""

WORKSPACE_FOLDER_NAME = "mlagent_workspace"
ARCHIVED_WORKSPACE_PREFIX = "mlagent_workspace_old_"
CONFIG_FILE_NAME = "config.yaml"

MEMORY_FOLDER_NAME = "memory"
LOGS_FOLDER_NAME = "logs"
ARTIFACTS_FOLDER_NAME = "artifacts"
REPORT_FOLDER_NAME = "report"
FEATURES_FOLDER_NAME = "features"
EVALUATION_FOLDER_NAME = "evaluation"
SPLITS_FILE_NAME = "splits.json"

LEDGER_FILE_NAME = "ledger.json"
PREVIOUS_LEDGER_FILE_NAME = "ledger.previous.json"
ENVIRONMENT_FILE_NAME = "environment.json"
PROGRESS_FILE_NAME = "progress.md"

RUN_LOG_FILE_NAME = "run.log"
LLM_CALLS_LOG_FILE_NAME = "llm_calls.jsonl"
COMMANDS_LOG_FILE_NAME = "commands.log"

# Folders the agents never list or read when they explore the project.
IGNORED_FOLDER_NAMES = [
    WORKSPACE_FOLDER_NAME,
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
]

MAX_CODE_ATTEMPTS = 5
MAX_ERROR_TEXT_CHARS = 1500
MAX_COMMAND_OUTPUT_CHARS = 250000
LOGGED_OUTPUT_CHARS = 2000
MAX_LIBRARY_LOOKUPS = 3
LIBRARY_HELP_CHARS = 1500

# A small safety net for shell commands written by the Coder. It is not a sandbox.
BLOCKED_SHELL_TEXT = [
    "rm -rf",
    "rmdir /s",
    "del /s",
    "remove-item -recurse",
    "mkfs",
    "format c:",
    "shutdown",
]

LLM_REPAIR_RETRIES = 2
LLM_CALL_TIMEOUT_SECONDS = 900
RETRY_WAIT_SECONDS = (5, 20, 60)
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

MAX_DATA_ENGINEER_ROUNDS = 3
BIG_FILE_MB = 200
MAX_FILES_IN_LISTING = 40
MAX_FACTS_CHARS = 6000
MAX_CHECKLIST_ITEMS = 12
MAX_QUESTIONS_PER_ROUND = 4

RANDOM_SEED = 42
DEFAULT_SPLIT_COUNT = 5
TUNE_TRIAL_COUNT = 20
TUNE_TOP_MODEL_COUNT = 2
TUNE_TIME_FRACTION = 0.6
MAX_FEATURE_SOURCE_CHARS = 5000
MAX_REVIEW_INSTRUCTIONS = 5
ENSEMBLE_MAX_RUNS = 3
PREDICTIONS_FILE_STEM = "predictions"
REPORT_FILE_NAME = "report.md"
