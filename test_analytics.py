"""
Test suite for the Program Integrity Analytics detectors and scoring.

Tests validate that each detector recovers the ground-truth FWA signals that
``data_gen`` injected (precision/recall sanity), that the composite risk score
ranks known-bad providers above clean ones, that recovery math is arithmetically
correct, and that nothing crashes on degenerate inputs (empty / single provider).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analytics as a
import data_gen
from data_gen import GenConfig


# A single shared, seeded dataset for the detection-quality tests.
@pytest.fixture(scope="module")
def claims() -> pd.DataFrame:
    return data_gen.generate_claims(GenConfig(seed=42))


@pytest.fixture(scope="module")
def truth(claims) -> pd.DataFrame:
    return data_gen.provider_ground_truth(claims)


def _providers_with(claims, arch):
    return set(claims.loc[claims["archetype"] == arch, "provider_id"])


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------

def test_dataset_is_reproducible():
    """Same seed -> identical dataset (needed for a data-file-free deploy)."""
    a1 = data_gen.generate_claims(GenConfig(seed=7))
    a2 = data_gen.generate_claims(GenConfig(seed=7))
    pd.testing.assert_frame_equal(a1, a2)


def test_dataset_shape_and_no_pii(claims):
    """Dataset is large, has the expected columns, and no obvious PII columns."""
    assert len(claims) > 50_000
    assert claims["provider_id"].nunique() == 800
    for col in ["claim_id", "provider_id", "beneficiary_id", "hcpcs_code",
                "units", "unit_cost", "billed_amount", "service_date"]:
        assert col in claims.columns
    # Synthetic IDs only; nothing name/ssn-like.
    assert not any(c.lower() in {"name", "ssn", "dob", "address"} for c in claims.columns)


def test_billed_amount_is_units_times_cost(claims):
    """Billed amount must equal units * unit_cost on every row."""
    expected = claims["units"] * claims["unit_cost"]
    assert np.allclose(claims["billed_amount"], expected)


# ---------------------------------------------------------------------------
# Detector recall: each detector catches its injected ground-truth signal
# ---------------------------------------------------------------------------

def test_detect_upcoding_catches_all_injected_upcoders(claims):
    """Every seeded upcoder is flagged (recall == 1.0)."""
    flagged = set(a.detect_upcoding(claims).query("flagged")["provider_id"])
    injected = _providers_with(claims, "upcoder")
    assert injected.issubset(flagged)


def test_detect_upcoding_precision_is_reasonable(claims):
    """Upcoding flags are dominated by true upcoders (few false positives)."""
    flagged = set(a.detect_upcoding(claims).query("flagged")["provider_id"])
    injected = _providers_with(claims, "upcoder")
    precision = len(flagged & injected) / len(flagged)
    assert precision >= 0.8


def test_detect_duplicates_catches_injected_dup_billers(claims):
    """Providers seeded as duplicate billers all appear among flagged duplicates."""
    dup = a.detect_duplicates(claims)
    flagged_providers = set(dup["provider_id"])
    injected = _providers_with(claims, "dup_biller")
    assert injected.issubset(flagged_providers)


def test_detect_duplicates_only_flags_copies_not_originals(claims):
    """Flagged duplicates equal (group size - 1) summed over groups."""
    dup = a.detect_duplicates(claims)
    # Reconstruct group sizes; flagged count per group should be size-1.
    cols = ["provider_id", "beneficiary_id", "hcpcs_code", "service_date"]
    sizes = claims.groupby(cols)["claim_id"].count()
    expected = int((sizes[sizes > 1] - 1).sum())
    assert len(dup) == expected


def test_detect_impossible_days_catches_injected(claims):
    """Every seeded impossible-day provider is flagged."""
    imp = a.detect_impossible_days(claims)
    flagged = set(imp["provider_id"])
    injected = _providers_with(claims, "impossible_day")
    assert injected.issubset(flagged)


def test_detect_impossible_days_all_exceed_24h(claims):
    """Every flagged provider-day genuinely exceeds 1,440 minutes."""
    imp = a.detect_impossible_days(claims)
    assert (imp["billed_minutes"] > a.MINUTES_PER_DAY).all()
    assert (imp["billed_hours"] > 24).all()


def test_detect_unit_outliers_catches_injected(claims):
    """Every seeded unit-abuser provider surfaces in the unit-outlier flags."""
    uo = a.detect_unit_outliers(claims)
    flagged = set(uo["provider_id"])
    injected = _providers_with(claims, "unit_abuser")
    assert injected.issubset(flagged)


def test_detect_unit_outliers_flagged_units_are_large(claims):
    """Flagged unit-outlier claims all carry more than one unit and positive z."""
    uo = a.detect_unit_outliers(claims)
    assert (uo["units"] > 1).all()
    assert (uo["unit_z"] >= a.UNIT_OUTLIER_Z_THRESHOLD).all()


def test_detect_beneficiary_volume_catches_injected(claims):
    """Every seeded volume abuser is flagged by the per-beneficiary detector."""
    vol = a.detect_beneficiary_volume(claims)
    flagged = set(vol.query("flagged")["provider_id"])
    injected = _providers_with(claims, "volume_abuser")
    assert injected.issubset(flagged)


# ---------------------------------------------------------------------------
# Risk scoring and ranking
# ---------------------------------------------------------------------------

def test_risk_score_ranks_bad_above_clean(claims, truth):
    """Mean risk of seeded bad actors far exceeds clean providers."""
    rs = a.provider_risk_score(claims).merge(truth[["provider_id", "archetype"]], on="provider_id")
    bad = rs[rs["archetype"] != "clean"]["risk_score"].mean()
    clean = rs[rs["archetype"] == "clean"]["risk_score"].mean()
    assert bad > clean * 5


def test_risk_score_top_providers_are_truly_bad(claims, truth):
    """The highest-risk providers are overwhelmingly real bad actors."""
    rs = a.provider_risk_score(claims).merge(truth[["provider_id", "archetype"]], on="provider_id")
    top = rs.sort_values("risk_score", ascending=False).head(40)
    bad_fraction = (top["archetype"] != "clean").mean()
    assert bad_fraction >= 0.9


def test_risk_score_bounds_and_coverage(claims):
    """Scores are within 0..100 and every provider receives a row."""
    rs = a.provider_risk_score(claims)
    assert rs["risk_score"].between(0, 100).all()
    assert rs["provider_id"].nunique() == claims["provider_id"].nunique()


# ---------------------------------------------------------------------------
# Recovery math
# ---------------------------------------------------------------------------

def test_recovery_duplicate_math_matches_billed_copies(claims):
    """Duplicate recovery per provider equals the billed amount of the copies."""
    rec = a.recovery_opportunity(claims).set_index("provider_id")
    dup = a.detect_duplicates(claims)
    by_prov = dup.groupby("provider_id")["billed_amount"].sum()
    for pid, val in by_prov.items():
        assert rec.loc[pid, "recovery_duplicate"] == pytest.approx(val)


def test_recovery_total_is_sum_of_components(claims):
    """total_recovery equals the sum of the four pattern components."""
    rec = a.recovery_opportunity(claims)
    comp = rec[["recovery_upcoding", "recovery_duplicate",
                "recovery_impossible", "recovery_units"]].sum(axis=1)
    assert np.allclose(rec["total_recovery"], comp)


def test_recovery_is_positive_and_below_total_billed(claims):
    """Dollars at risk are positive but a sane fraction of total billed."""
    rec = a.recovery_opportunity(claims)
    total_billed = claims["billed_amount"].sum()
    assert rec["total_recovery"].sum() > 0
    assert rec["total_recovery"].sum() < total_billed


# ---------------------------------------------------------------------------
# Worklist + KPIs
# ---------------------------------------------------------------------------

def test_worklist_is_ranked_by_priority(claims):
    """Worklist priority score is monotonically non-increasing and ranks start at 1."""
    wl = a.investigator_worklist(claims)
    assert wl["priority_rank"].iloc[0] == 1
    assert (wl["priority_score"].values[:-1] >= wl["priority_score"].values[1:]).all()
    # Every row has a stated reason.
    assert (wl["reasons"].str.len() > 0).all()


def test_worklist_top_providers_overlap_known_bad(claims, truth):
    """Top of the worklist is populated by seeded bad actors."""
    wl = a.investigator_worklist(claims).merge(truth[["provider_id", "archetype"]], on="provider_id")
    top = wl.head(20)
    assert (top["archetype"] != "clean").mean() >= 0.9


def test_summary_kpis_fields_and_values(claims):
    """KPIs expose the expected keys with internally consistent values."""
    k = a.summary_kpis(claims)
    for key in ["total_billed", "dollars_at_risk", "pct_at_risk", "n_providers",
                "n_high_risk_providers", "pct_claims_flagged", "n_claims"]:
        assert key in k
    assert k["n_providers"] == 800
    assert 0 <= k["pct_at_risk"] <= 100
    assert 0 <= k["pct_claims_flagged"] <= 100
    assert k["dollars_at_risk"] == pytest.approx(
        a.recovery_opportunity(claims)["total_recovery"].sum()
    )


# ---------------------------------------------------------------------------
# Edge cases: must not crash
# ---------------------------------------------------------------------------

def test_empty_input_does_not_crash():
    """All public functions return empty structures (not errors) on empty input."""
    empty = data_gen.generate_claims(GenConfig(seed=1)).iloc[0:0]
    assert a.detect_upcoding(empty).empty
    assert a.detect_duplicates(empty).empty
    assert a.detect_impossible_days(empty).empty
    assert a.detect_unit_outliers(empty).empty
    assert a.recovery_opportunity(empty).empty
    assert a.provider_risk_score(empty).empty
    assert a.investigator_worklist(empty).empty
    k = a.summary_kpis(empty)
    assert k["total_billed"] == 0.0
    assert k["n_providers"] == 0


def test_single_provider_does_not_crash():
    """A one-provider dataset runs end-to-end without error."""
    full = data_gen.generate_claims(GenConfig(seed=3))
    one = full[full["provider_id"] == full["provider_id"].iloc[0]].copy()
    # Peer-relative detectors should yield no flags (no peers to compare to).
    assert not a.detect_upcoding(one)["flagged"].any()
    rs = a.provider_risk_score(one)
    assert len(rs) == 1
    assert rs["risk_score"].between(0, 100).all()
    # Worklist and KPIs still compute.
    a.investigator_worklist(one)
    k = a.summary_kpis(one)
    assert k["n_providers"] == 1
