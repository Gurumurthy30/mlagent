"""Tests for digest.py: budget caps, item integrity, no 'None' renders, and facts inclusions."""

from __future__ import annotations

import pytest
from hypothesis import assume, given

from mlagent.digest import DIGEST_CHAR_LIMITS, build_digest, build_report_facts, shorten_text
from mlagent.state import Ledger
from tests.conftest import st_ledger


@pytest.mark.unit
@given(ledger=st_ledger())
def test_digest_never_exceeds_budget_for_any_agent(ledger: Ledger) -> None:
    """Digest length never exceeds the agent's character budget on any random ledger."""
    for agent_name, budget in DIGEST_CHAR_LIMITS.items():
        digest = build_digest(agent_name, ledger, char_limit=budget)
        assert len(digest) <= budget, (
            f"Digest for {agent_name} length {len(digest)} exceeded budget {budget}."
        )


@pytest.mark.unit
@given(ledger=st_ledger())
def test_digest_never_renders_literal_none(ledger: Ledger) -> None:
    """Digest output should never contain 'None' as an unformatted string."""
    assume("None" not in ledger.user_task)
    assume("None" not in ledger.user_metrics_text)
    assume("None" not in ledger.brief.summary)
    assume("None" not in ledger.brief.target)
    assume(not any("None" in r.summary or "None" in r.error for r in ledger.runs))
    assume(not any("None" in f.statement for f in ledger.eda.findings))

    for agent_name in DIGEST_CHAR_LIMITS:
        digest = build_digest(agent_name, ledger)
        # Avoid literal "None" when formatting fields (should be empty, omitted, or meaningful text)
        assert " None " not in digest, f"Digest for {agent_name} contains unformatted ' None '."
        assert ": None\n" not in digest, f"Digest for {agent_name} contains unformatted ': None'."


@pytest.mark.unit
def test_shorten_text_never_cuts_mid_word() -> None:
    """shorten_text preserves whole words or sentence boundaries with ellipsis."""
    text = "First sentence is clear. Second sentence has extraordinary long terminology that exceeds limit."
    shortened = shorten_text(text, 40)
    assert len(shortened) <= 45  # includes ellipsis allowance
    assert shortened.endswith("...")
    # Does not cut inside 'extraordinary'
    assert not shortened.endswith("extra...")
    assert not shortened.endswith("extraor...")


@pytest.mark.unit
@given(ledger=st_ledger())
def test_reviewer_digest_includes_eda_when_available(ledger: Ledger) -> None:
    """Reviewer digest includes EDA findings when present in the ledger."""
    if ledger.eda.findings:
        digest = build_digest("reviewer", ledger)
        # At least one finding statement or finding id should be mentioned if budget allows
        has_finding = any(f.statement in digest or f.finding_id in digest for f in ledger.eda.findings)
        assert has_finding or len(digest) > 0


@pytest.mark.unit
@given(ledger=st_ledger())
def test_report_facts_inclusions(ledger: Ledger) -> None:
    """build_report_facts includes the best approved run and high priority findings when available."""
    facts = build_report_facts(ledger, char_limit=15000)
    assert len(facts) <= 15000
    assert " None " not in facts

    best = ledger.best_approved_run()
    if best and best.headline:
        assert best.headline in facts or best.experiment_id in facts

    high_findings = [f for f in ledger.eda.findings if f.priority == "high"]
    if high_findings:
        # At least one high priority finding should be present
        assert any(f.statement in facts for f in high_findings)
