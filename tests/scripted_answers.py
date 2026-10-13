"""Builds the scripted answers that agent tests feed to the fake LLM.

An agent that calls the Coder uses the LLM twice per step: once for the Coder (code),
once for itself (JSON). Tests list the answers in the same order.
"""


def coder_answer_printing(tag: str, data: dict) -> str:
    """A Coder answer: a script that prints the data between <TAG> and </TAG>."""
    script = f"import json\nprint('<{tag}>' + json.dumps({data!r}) + '</{tag}>')"
    return f"```python\n{script}\n```"


def coder_answer_failing(error_message: str) -> str:
    """A Coder answer: a script that always raises the same error."""
    return f"```python\nraise RuntimeError({error_message!r})\n```"


def two_failed_coder_attempts(error_message: str) -> list[str]:
    """The same error twice in a row makes the Coder give up after two attempts."""
    return [coder_answer_failing(error_message), coder_answer_failing(error_message)]
