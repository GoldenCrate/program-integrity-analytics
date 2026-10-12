"""
Program Integrity Analytics — FWA detection and investigator prioritization.

Framework-free analytics (pure pandas / numpy, no Streamlit imports) so the
logic is unit-testable and reusable. Each public function documents the
business question it answers.

Detection philosophy: no single flag is proof of fraud. Each detector surfaces
a *statistical anomaly* relative to a peer baseline (same specialty) or an
operational impossibility. The composite ``provider_risk_score`` combines the
signals, and ``investigator_worklist`` ranks providers by risk x dollars so a
limited investigative team works the highest-value leads first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_gen import HIGH_COST_CODES

# Minutes in a 24-hour day; the ceiling for a single provider's billed time.
MINUTES_PER_DAY = 24 * 60

# Default statistical thresholds (documented, tunable).
UPCODING_Z_THRESHOLD = 2.0       # std-devs above specialty peer mean
UNIT_OUTLIER_Z_THRESHOLD = 4.0   # std-devs above per-code unit mean
VOLUME_Z_THRESHOLD = 2.5         # std-devs above specialty peer claims/beneficiary


# ----------------------------------------------------------------------------
# Individual detectors
# ----------------------------------------------------------------------------

def detect_upcoding(claims: pd.DataFrame, z_threshold: float = UPCODING_Z_THRESHOLD) -> pd.DataFrame:
    """Flag providers who bill high-cost codes far more than specialty peers.

    Business question: which providers systematically select the most expensive
    procedure codes within their specialty at an abnormal rate (upcoding)?

    Method: for each provider compute the share of claims that fall on the two
    highest-cost codes in their specialty. Within each specialty, z-score that
    share against peers. Flag providers whose z-score exceeds ``z_threshold``.

    Returns one row per provider with the high-cost share, peer z-score, and a
    boolean ``flagged``.
    """
    if claims.empty:
        return pd.DataFrame(
            columns=["provider_id", "specialty", "high_cost_share", "peer_z", "flagged"]
        )

    df = claims.copy()
    df["is_high_cost"] = [
        code in HIGH_COST_CODES.get(spec, set())
        for code, spec in zip(df["hcpcs_code"], df["specialty"])
    ]

    grp = (
        df.groupby(["provider_id", "specialty"])
        .agg(high_cost_share=("is_high_cost", "mean"), n_claims=("claim_id", "count"))
        .reset_index()
    )

    # Peer baseline within each specialty.
    grp["peer_mean"] = grp.groupby("specialty")["high_cost_share"].transform("mean")
    grp["peer_std"] = grp.groupby("specialty")["high_cost_share"].transform("std").fillna(0.0)
    grp["peer_z"] = np.where(
        grp["peer_std"] > 0,
        (grp["high_cost_share"] - grp["peer_mean"]) / grp["peer_std"],
        0.0,
    )
    grp["flagged"] = grp["peer_z"] >= z_threshold
    return grp[["provider_id", "specialty", "high_cost_share", "peer_z", "flagged"]]


def detect_duplicates(claims: pd.DataFrame) -> pd.DataFrame:
    """Flag duplicate / near-duplicate claim lines.

    Business question: is the same service being billed more than once (same
    provider, beneficiary, procedure code, and service date)?

    Method: group on (provider, beneficiary, code, date). Any group with more
    than one claim is a duplicate set; every claim after the first is flagged
    as a duplicate. Returns the flagged duplicate claim rows (not the originals)
    with the size of the duplicate group.
    """
    cols = ["provider_id", "beneficiary_id", "hcpcs_code", "service_date"]
    if claims.empty:
        return claims.iloc[0:0].assign(dup_group_size=pd.Series(dtype=int))

    df = claims.sort_values("claim_id").copy()
    df["dup_group_size"] = df.groupby(cols)["claim_id"].transform("count")
    df["dup_rank"] = df.groupby(cols).cumcount()  # 0 == original
    flagged = df[(df["dup_group_size"] > 1) & (df["dup_rank"] > 0)].copy()
    return flagged.drop(columns=["dup_rank"])


def detect_impossible_days(claims: pd.DataFrame, minutes_cap: int = MINUTES_PER_DAY) -> pd.DataFrame:
    """Flag provider-days with more billed service time than exists in a day.

    Business question: does any provider bill more than 24 hours of service in a
    single calendar day (an operational impossibility)?

    Method: sum ``duration_min * units`` per provider per service date. Flag
    provider-days exceeding ``minutes_cap`` (default 1,440 minutes = 24h).
    Returns one row per flagged provider-day with total billed minutes.
    """
    if claims.empty:
        return pd.DataFrame(
            columns=["provider_id", "service_date", "billed_minutes", "billed_hours", "n_claims"]
        )

    df = claims.copy()
    df["claim_minutes"] = df["duration_min"] * df["units"]
    daily = (
        df.groupby(["provider_id", "service_date"])
        .agg(billed_minutes=("claim_minutes", "sum"), n_claims=("claim_id", "count"))
        .reset_index()
    )
    flagged = daily[daily["billed_minutes"] > minutes_cap].copy()
    flagged["billed_hours"] = flagged["billed_minutes"] / 60.0
    return flagged[["provider_id", "service_date", "billed_minutes", "billed_hours", "n_claims"]]


def detect_unit_outliers(claims: pd.DataFrame, z_threshold: float = UNIT_OUTLIER_Z_THRESHOLD) -> pd.DataFrame:
    """Flag individual claims with an abnormal number of billed units.

    Business question: which single claims bill far more units of a procedure
    than is normal for that procedure code?

    Method: within each HCPCS code, z-score the unit count. Flag claims whose
    units exceed the per-code mean by more than ``z_threshold`` std-devs (and at
    least a floor of >1 unit, so noise on single-unit codes is ignored).
    Returns the flagged claim rows with the computed z-score.
    """
    if claims.empty:
        return claims.iloc[0:0].assign(unit_z=pd.Series(dtype=float))

    df = claims.copy()
    df["unit_mean"] = df.groupby("hcpcs_code")["units"].transform("mean")
    df["unit_std"] = df.groupby("hcpcs_code")["units"].transform("std").fillna(0.0)
    df["unit_z"] = np.where(
        df["unit_std"] > 0,
        (df["units"] - df["unit_mean"]) / df["unit_std"],
        0.0,
    )
    flagged = df[(df["unit_z"] >= z_threshold) & (df["units"] > 1)].copy()
    return flagged.drop(columns=["unit_mean", "unit_std"])


def detect_beneficiary_volume(claims: pd.DataFrame, z_threshold: float = VOLUME_Z_THRESHOLD) -> pd.DataFrame:
    """Flag providers funneling abnormal claim volume through few beneficiaries.

    Business question: which providers bill an unusually high number of claims
    per beneficiary compared to specialty peers (a classic volume-abuse / phantom
    patient pattern)?

    Method: compute claims-per-beneficiary per provider, then z-score within
    specialty. Flag providers above ``z_threshold``.
    """
    if claims.empty:
        return pd.DataFrame(
            columns=["provider_id", "specialty", "claims_per_bene", "peer_z", "flagged"]
        )

    grp = (
        claims.groupby(["provider_id", "specialty"])
        .agg(n_claims=("claim_id", "count"), n_benes=("beneficiary_id", "nunique"))
        .reset_index()
    )
    grp["claims_per_bene"] = grp["n_claims"] / grp["n_benes"]
    grp["peer_mean"] = grp.groupby("specialty")["claims_per_bene"].transform("mean")
    grp["peer_std"] = grp.groupby("specialty")["claims_per_bene"].transform("std").fillna(0.0)
    grp["peer_z"] = np.where(
        grp["peer_std"] > 0,
        (grp["claims_per_bene"] - grp["peer_mean"]) / grp["peer_std"],
        0.0,
    )
    grp["flagged"] = grp["peer_z"] >= z_threshold
    return grp[["provider_id", "specialty", "claims_per_bene", "peer_z", "flagged"]]


# ----------------------------------------------------------------------------
# Recovery / dollars at risk
# ----------------------------------------------------------------------------

def recovery_opportunity(claims: pd.DataFrame) -> pd.DataFrame:
    """Estimate recoverable dollars per provider, by FWA pattern.

    Business question: of the money billed, how much is plausibly recoverable,
    and from which providers — so investigators prioritize by dollars at risk?

    Estimation rules (deliberately conservative, documented assumptions):
      * Upcoding: for flagged upcoders, the excess of billed amount over the
        specialty *median* claim amount, summed across their claims (only the
        positive excess counts).
      * Duplicates: the full billed amount of every duplicate claim line (the
        original is legitimate; the copies are not).
      * Impossible days: the billed amount on flagged provider-days scaled by the
        fraction of billed time that exceeds 24h (the physically impossible part).
      * Unit outliers: the billed amount attributable to units above the per-code
        mean (excess units x unit cost).

    Returns one row per provider that has any recovery opportunity, with a
    per-pattern breakdown and a ``total_recovery`` column.
    """
    if claims.empty:
        return pd.DataFrame(
            columns=[
                "provider_id", "recovery_upcoding", "recovery_duplicate",
                "recovery_impossible", "recovery_units", "total_recovery",
            ]
        )

    providers = claims["provider_id"].unique()
    rec = pd.DataFrame({"provider_id": providers}).set_index("provider_id")
    rec["recovery_upcoding"] = 0.0
    rec["recovery_duplicate"] = 0.0
    rec["recovery_impossible"] = 0.0
    rec["recovery_units"] = 0.0

    # --- Upcoding: excess over specialty median claim amount for flagged upcoders
    upc = detect_upcoding(claims)
    flagged_upcoders = set(upc.loc[upc["flagged"], "provider_id"])
    if flagged_upcoders:
        spec_median = claims.groupby("specialty")["billed_amount"].median()
        sub = claims[claims["provider_id"].isin(flagged_upcoders)].copy()
        sub["excess"] = (sub["billed_amount"] - sub["specialty"].map(spec_median)).clip(lower=0)
        by_prov = sub.groupby("provider_id")["excess"].sum()
        rec.loc[by_prov.index, "recovery_upcoding"] = by_prov.values

    # --- Duplicates: full billed amount of duplicate copies
    dup = detect_duplicates(claims)
    if not dup.empty:
        by_prov = dup.groupby("provider_id")["billed_amount"].sum()
        rec.loc[by_prov.index, "recovery_duplicate"] = by_prov.values

    # --- Impossible days: billed amount scaled by the impossible time fraction
    imp = detect_impossible_days(claims)
    if not imp.empty:
        df = claims.copy()
        df["claim_minutes"] = df["duration_min"] * df["units"]
        day_bill = (
            df.groupby(["provider_id", "service_date"])
            .agg(billed_amount=("billed_amount", "sum"), billed_minutes=("claim_minutes", "sum"))
            .reset_index()
        )
        merged = imp.merge(day_bill, on=["provider_id", "service_date"], how="left")
        merged["impossible_frac"] = (
            (merged["billed_minutes_y"] - MINUTES_PER_DAY) / merged["billed_minutes_y"]
        ).clip(lower=0)
        merged["recovery"] = merged["billed_amount"] * merged["impossible_frac"]
        by_prov = merged.groupby("provider_id")["recovery"].sum()
        rec.loc[by_prov.index, "recovery_impossible"] = by_prov.values

    # --- Unit outliers: excess units above per-code mean x unit cost
    units = detect_unit_outliers(claims)
    if not units.empty:
        code_mean = claims.groupby("hcpcs_code")["units"].mean()
        u = units.copy()
        u["excess_units"] = (u["units"] - u["hcpcs_code"].map(code_mean)).clip(lower=0)
        u["recovery"] = u["excess_units"] * u["unit_cost"]
        by_prov = u.groupby("provider_id")["recovery"].sum()
        rec.loc[by_prov.index, "recovery_units"] = by_prov.values

    rec["total_recovery"] = rec[
        ["recovery_upcoding", "recovery_duplicate", "recovery_impossible", "recovery_units"]
    ].sum(axis=1)

    rec = rec[rec["total_recovery"] > 0].reset_index()
    return rec.sort_values("total_recovery", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Composite risk score
# ----------------------------------------------------------------------------

# Documented weighting of the composite risk score (sums to 1.0). Upcoding and
# duplicates carry the most weight because they are the clearest intentional
# overbilling patterns; impossible days are a hard operational red flag.
RISK_WEIGHTS = {
    "upcoding": 0.25,
    "duplicate": 0.20,
    "impossible": 0.25,
    "units": 0.15,
    "volume": 0.15,
}


def _minmax(s: pd.Series) -> pd.Series:
    """Scale a series to 0..1; constant series -> all zeros."""
    lo, hi = s.min(), s.max()
    if hi <= lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def provider_risk_score(claims: pd.DataFrame) -> pd.DataFrame:
    """Compute a 0-100 composite FWA risk score for every provider.

    Business question: combining all signals, how risky is each provider overall
    so the team can triage the whole provider population with one number?

    Method: build a per-provider feature for each signal, min-max scale each
    feature to 0..1 across providers, take a documented weighted sum
    (``RISK_WEIGHTS``), and rescale to 0..100. Also returns the raw signal
    components and a human-readable list of fired reasons.

    Returns one row per provider sorted by ``risk_score`` descending.
    """
    if claims.empty:
        return pd.DataFrame(
            columns=["provider_id", "specialty", "risk_score", "reasons"]
        )

    base = (
        claims.groupby(["provider_id", "specialty"])
        .agg(n_claims=("claim_id", "count"), total_billed=("billed_amount", "sum"))
        .reset_index()
        .set_index("provider_id")
    )

    # --- Signal features ---
    upc = detect_upcoding(claims).set_index("provider_id")
    base["upcoding_z"] = upc["peer_z"].reindex(base.index).clip(lower=0).fillna(0.0)
    base["upcoding_flag"] = upc["flagged"].reindex(base.index).fillna(False)

    dup = detect_duplicates(claims)
    dup_counts = dup.groupby("provider_id")["claim_id"].count() if not dup.empty else pd.Series(dtype=int)
    base["dup_count"] = dup_counts.reindex(base.index).fillna(0.0)

    imp = detect_impossible_days(claims)
    imp_counts = imp.groupby("provider_id")["service_date"].count() if not imp.empty else pd.Series(dtype=int)
    base["impossible_days"] = imp_counts.reindex(base.index).fillna(0.0)

    units = detect_unit_outliers(claims)
    unit_counts = units.groupby("provider_id")["claim_id"].count() if not units.empty else pd.Series(dtype=int)
    base["unit_outliers"] = unit_counts.reindex(base.index).fillna(0.0)

    vol = detect_beneficiary_volume(claims).set_index("provider_id")
    base["volume_z"] = vol["peer_z"].reindex(base.index).clip(lower=0).fillna(0.0)
    base["volume_flag"] = vol["flagged"].reindex(base.index).fillna(False)

    # --- Scale and combine ---
    comp = (
        RISK_WEIGHTS["upcoding"] * _minmax(base["upcoding_z"])
        + RISK_WEIGHTS["duplicate"] * _minmax(base["dup_count"])
        + RISK_WEIGHTS["impossible"] * _minmax(base["impossible_days"])
        + RISK_WEIGHTS["units"] * _minmax(base["unit_outliers"])
        + RISK_WEIGHTS["volume"] * _minmax(base["volume_z"])
    )
    base["risk_score"] = (comp * 100).round(1)

    # --- Human-readable reasons ---
    def _reasons(row: pd.Series) -> str:
        out = []
        if row["upcoding_flag"]:
            out.append("Upcoding (high-cost code over-billing)")
        if row["dup_count"] > 0:
            out.append(f"Duplicate claims ({int(row['dup_count'])})")
        if row["impossible_days"] > 0:
            out.append(f"Impossible day(s) >24h ({int(row['impossible_days'])})")
        if row["unit_outliers"] > 0:
            out.append(f"Excessive units ({int(row['unit_outliers'])} claims)")
        if row["volume_flag"]:
            out.append("Abnormal claims per beneficiary")
        return "; ".join(out) if out else "No signals fired"

    base["reasons"] = base.apply(_reasons, axis=1)

    out = base.reset_index()[
        [
            "provider_id", "specialty", "risk_score", "reasons",
            "n_claims", "total_billed", "upcoding_z", "dup_count",
            "impossible_days", "unit_outliers", "volume_z",
        ]
    ]
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Investigator worklist — the key deliverable
# ----------------------------------------------------------------------------

def investigator_worklist(claims: pd.DataFrame, risk_threshold: float = 0.0) -> pd.DataFrame:
    """Rank providers to investigate by risk x dollars at risk.

    Business question: given limited investigator capacity, which providers
    should be worked first to recover the most money from the riskiest actors?

    Method: join the composite risk score to estimated recoverable dollars, and
    sort by a priority score = ``risk_score * dollars_at_risk`` (so a provider
    must be both risky *and* expensive to rise to the top). Providers with no
    risk signals are dropped. ``risk_threshold`` optionally filters low scores.

    Returns the ranked worklist with risk score, dollars at risk, the fired
    reasons, and a 1-based ``priority_rank``.
    """
    if claims.empty:
        return pd.DataFrame(
            columns=[
                "priority_rank", "provider_id", "specialty", "risk_score",
                "dollars_at_risk", "priority_score", "reasons",
            ]
        )

    risk = provider_risk_score(claims)
    rec = recovery_opportunity(claims)[["provider_id", "total_recovery"]]

    work = risk.merge(rec, on="provider_id", how="left")
    work["dollars_at_risk"] = work["total_recovery"].fillna(0.0)

    # Keep providers that are either scored risky or carry dollars at risk.
    work = work[(work["risk_score"] > risk_threshold) | (work["dollars_at_risk"] > 0)].copy()
    work = work[work["reasons"] != "No signals fired"].copy()

    work["priority_score"] = work["risk_score"] * work["dollars_at_risk"]
    work = work.sort_values(
        ["priority_score", "risk_score", "dollars_at_risk"], ascending=False
    ).reset_index(drop=True)
    work["priority_rank"] = work.index + 1

    return work[
        [
            "priority_rank", "provider_id", "specialty", "risk_score",
            "dollars_at_risk", "priority_score", "reasons",
        ]
    ]


# ----------------------------------------------------------------------------
# Summary KPIs
# ----------------------------------------------------------------------------

def summary_kpis(claims: pd.DataFrame, risk_threshold: float = 20.0) -> dict:
    """Compute program-level KPIs for the executive summary.

    Business question: at a glance, how big is the book of business, how much is
    at risk, and how concentrated is the problem?

    Returns a dict of headline numbers: total billed, dollars at risk, % of
    dollars at risk, number of providers, providers above the risk threshold,
    and the share of claims touched by any FWA signal.
    """
    if claims.empty:
        return {
            "total_billed": 0.0, "dollars_at_risk": 0.0, "pct_at_risk": 0.0,
            "n_providers": 0, "n_high_risk_providers": 0, "pct_claims_flagged": 0.0,
            "n_claims": 0,
        }

    total_billed = float(claims["billed_amount"].sum())
    rec = recovery_opportunity(claims)
    dollars_at_risk = float(rec["total_recovery"].sum()) if not rec.empty else 0.0

    risk = provider_risk_score(claims)
    n_high_risk = int((risk["risk_score"] >= risk_threshold).sum())

    # Share of claim lines touched by any detector.
    flagged_ids: set = set()
    dup = detect_duplicates(claims)
    if not dup.empty:
        flagged_ids |= set(dup["claim_id"])
    units = detect_unit_outliers(claims)
    if not units.empty:
        flagged_ids |= set(units["claim_id"])
    imp = detect_impossible_days(claims)
    if not imp.empty:
        imp_keys = set(zip(imp["provider_id"], imp["service_date"]))
        mask = [
            (p, d) in imp_keys
            for p, d in zip(claims["provider_id"], claims["service_date"])
        ]
        flagged_ids |= set(claims.loc[mask, "claim_id"])

    pct_claims_flagged = 100.0 * len(flagged_ids) / len(claims)

    return {
        "total_billed": total_billed,
        "dollars_at_risk": dollars_at_risk,
        "pct_at_risk": 100.0 * dollars_at_risk / total_billed if total_billed else 0.0,
        "n_providers": int(claims["provider_id"].nunique()),
        "n_high_risk_providers": n_high_risk,
        "pct_claims_flagged": pct_claims_flagged,
        "n_claims": int(len(claims)),
    }
