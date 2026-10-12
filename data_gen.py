"""
Synthetic claims data generator for the Program Integrity Analytics demo.

This module fabricates a realistic-looking Medicare/Medicaid claims dataset and
INTENTIONALLY injects fraud, waste, and abuse (FWA) signals so the analytics in
``analytics.py`` have something concrete to detect. Every record is synthetic:
there is no real patient data, no PII, and no real provider identifiers. Ground
truth flags (columns prefixed ``gt_``) record exactly which claims were seeded
as anomalous so the test suite can measure detection precision/recall.

The generator is fully seeded (``np.random.seed`` / ``random.seed``) so the same
dataset is produced on every run, which is what lets the Streamlit app build its
data on launch without shipping any data files.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Reference data: specialties, HCPCS-like codes, regions
# ----------------------------------------------------------------------------

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]

# Each specialty maps to a list of (code, unit_cost, typical_minutes).
# Codes are ordered from lowest to highest intensity so the last two in each
# list are the "high-cost" codes that upcoders over-bill.
SPECIALTY_CODES: dict[str, list[tuple[str, float, int]]] = {
    "Internal Medicine": [
        ("G0402", 55.0, 15),
        ("99213", 95.0, 20),
        ("99214", 140.0, 30),
        ("99215", 210.0, 40),
        ("99497", 300.0, 60),
    ],
    "Cardiology": [
        ("93000", 30.0, 10),
        ("93306", 230.0, 45),
        ("93458", 1800.0, 90),
        ("92928", 9500.0, 150),
        ("33533", 21000.0, 240),
    ],
    "Orthopedics": [
        ("73610", 45.0, 15),
        ("20610", 90.0, 20),
        ("29881", 3200.0, 75),
        ("27447", 16500.0, 180),
        ("27130", 19500.0, 210),
    ],
    "Physical Therapy": [
        ("97110", 40.0, 15),
        ("97112", 48.0, 15),
        ("97140", 52.0, 15),
        ("97530", 60.0, 20),
        ("97535", 72.0, 30),
    ],
    "Radiology": [
        ("71045", 35.0, 10),
        ("74176", 420.0, 30),
        ("70553", 1200.0, 45),
        ("74178", 1500.0, 50),
        ("78452", 2100.0, 60),
    ],
    "Dermatology": [
        ("99212", 70.0, 10),
        ("11102", 120.0, 15),
        ("17000", 180.0, 20),
        ("11640", 650.0, 40),
        ("14060", 1400.0, 60),
    ],
}

SPECIALTIES = list(SPECIALTY_CODES.keys())


@dataclass
class GenConfig:
    """Tunable sizes for the synthetic dataset."""

    n_providers: int = 800
    n_beneficiaries: int = 15_000
    n_claims: int = 60_000
    months: int = 24
    seed: int = 42

    # Fraction of providers seeded with each FWA archetype (disjoint groups).
    frac_upcoders: float = 0.05
    frac_dup_billers: float = 0.04
    frac_impossible: float = 0.03
    frac_unit_abusers: float = 0.03
    frac_volume_abusers: float = 0.03


# Convenience lookup: (specialty, code) -> (unit_cost, minutes)
def _code_lookup() -> dict[tuple[str, str], tuple[float, int]]:
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for spec, codes in SPECIALTY_CODES.items():
        for code, cost, minutes in codes:
            out[(spec, code)] = (cost, minutes)
    return out


CODE_LOOKUP = _code_lookup()


def _high_cost_codes() -> dict[str, set[str]]:
    """Return the set of 'high-cost' codes (top two by cost) per specialty."""
    out: dict[str, set[str]] = {}
    for spec, codes in SPECIALTY_CODES.items():
        ordered = sorted(codes, key=lambda c: c[1])
        out[spec] = {ordered[-1][0], ordered[-2][0]}
    return out


HIGH_COST_CODES = _high_cost_codes()


def generate_providers(cfg: GenConfig) -> pd.DataFrame:
    """Build the provider dimension with seeded FWA archetype labels.

    Business question: which providers exist, what do they practice, and
    (ground truth) which ones were planted as bad actors of each type?
    """
    rng = np.random.default_rng(cfg.seed)
    ids = [f"P{idx:05d}" for idx in range(1, cfg.n_providers + 1)]
    specialties = rng.choice(SPECIALTIES, size=cfg.n_providers)
    regions = rng.choice(REGIONS, size=cfg.n_providers)

    providers = pd.DataFrame(
        {
            "provider_id": ids,
            "specialty": specialties,
            "region": regions,
            "archetype": "clean",
        }
    )

    # Assign disjoint archetype groups.
    order = rng.permutation(cfg.n_providers)
    cursor = 0

    def _take(frac: float) -> np.ndarray:
        nonlocal cursor
        n = int(round(frac * cfg.n_providers))
        picked = order[cursor : cursor + n]
        cursor += n
        return picked

    for frac, label in [
        (cfg.frac_upcoders, "upcoder"),
        (cfg.frac_dup_billers, "dup_biller"),
        (cfg.frac_impossible, "impossible_day"),
        (cfg.frac_unit_abusers, "unit_abuser"),
        (cfg.frac_volume_abusers, "volume_abuser"),
    ]:
        idx = _take(frac)
        providers.loc[idx, "archetype"] = label

    return providers


def generate_claims(cfg: GenConfig | None = None, seed: int | None = None) -> pd.DataFrame:
    """Generate the full synthetic claims table with injected FWA signals.

    Business question: produce a claims dataset that looks operationally real
    and contains known, labeled fraud/waste/abuse patterns so detection logic
    can be built and validated.

    Returns a DataFrame of individual claim lines with cost, timing, and
    ground-truth (``gt_*``) flags. One row == one claim line.
    """
    if cfg is None:
        cfg = GenConfig()
    if seed is not None:
        cfg = GenConfig(**{**cfg.__dict__, "seed": seed})

    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    providers = generate_providers(cfg)
    prov_by_arch = {
        arch: providers.loc[providers["archetype"] == arch, "provider_id"].tolist()
        for arch in providers["archetype"].unique()
    }
    prov_meta = providers.set_index("provider_id")[["specialty", "region"]].to_dict("index")

    beneficiaries = np.array([f"B{idx:06d}" for idx in range(1, cfg.n_beneficiaries + 1)])

    start = np.datetime64("2023-01-01")
    day_span = cfg.months * 30

    records: list[dict] = []
    claim_counter = 0

    def _next_claim_id() -> str:
        nonlocal claim_counter
        claim_counter += 1
        return f"C{claim_counter:08d}"

    # ---- allocate claim counts per provider (clean providers get the bulk) ----
    # Give each provider a baseline volume; bad actors get a modest boost so the
    # anomalies are not purely volume-driven (that would be too easy to detect).
    base_counts = rng.integers(30, 120, size=cfg.n_providers)
    provider_counts = dict(zip(providers["provider_id"], base_counts))

    def _pick_code(specialty: str, upcode: bool) -> str:
        codes = SPECIALTY_CODES[specialty]
        names = [c[0] for c in codes]
        if upcode:
            # Heavily weight the two highest-cost codes.
            weights = np.array([1, 1, 2, 8, 12], dtype=float)[: len(names)]
        else:
            # Normal practice leans to low/mid intensity codes.
            weights = np.array([10, 9, 6, 2, 1], dtype=float)[: len(names)]
        weights = weights / weights.sum()
        return str(rng.choice(names, p=weights))

    def _service_date() -> np.datetime64:
        return start + np.timedelta64(int(rng.integers(0, day_span)), "D")

    def _base_record(provider_id: str) -> dict:
        meta = prov_meta[provider_id]
        bene = str(rng.choice(beneficiaries))
        return {
            "claim_id": _next_claim_id(),
            "provider_id": provider_id,
            "beneficiary_id": bene,
            "specialty": meta["specialty"],
            "region": meta["region"],
        }

    def _finalize(rec: dict, code: str, units: int, service_date: np.datetime64,
                  gt_upcoding=False, gt_duplicate=False,
                  gt_impossible_day=False, gt_unit_outlier=False) -> dict:
        spec = rec["specialty"]
        unit_cost, minutes = CODE_LOOKUP[(spec, code)]
        rec.update(
            {
                "hcpcs_code": code,
                "units": int(units),
                "unit_cost": float(unit_cost),
                "duration_min": int(minutes),
                "service_date": pd.Timestamp(service_date),
                "billed_amount": float(unit_cost) * int(units),
                "gt_upcoding": bool(gt_upcoding),
                "gt_duplicate": bool(gt_duplicate),
                "gt_impossible_day": bool(gt_impossible_day),
                "gt_unit_outlier": bool(gt_unit_outlier),
            }
        )
        return rec

    # ---- 1. Normal / clean claims for every provider -------------------------
    for provider_id, count in provider_counts.items():
        arch = providers.set_index("provider_id").loc[provider_id, "archetype"]
        spec = prov_meta[provider_id]["specialty"]
        # Bad actors still have a majority of ordinary-looking claims.
        for _ in range(int(count)):
            upcode = arch == "upcoder" and rng.random() < 0.55
            code = _pick_code(spec, upcode=upcode)
            units = int(rng.integers(1, 4))
            rec = _base_record(provider_id)
            records.append(
                _finalize(rec, code, units, _service_date(), gt_upcoding=upcode)
            )

    # ---- 2. Duplicate / near-duplicate claims --------------------------------
    for provider_id in prov_by_arch.get("dup_biller", []):
        spec = prov_meta[provider_id]["specialty"]
        n_groups = int(rng.integers(8, 16))
        for _ in range(n_groups):
            code = _pick_code(spec, upcode=False)
            units = int(rng.integers(1, 3))
            svc = _service_date()
            bene = str(rng.choice(beneficiaries))
            copies = int(rng.integers(2, 5))  # original + duplicates
            for c in range(copies):
                rec = {
                    "claim_id": _next_claim_id(),
                    "provider_id": provider_id,
                    "beneficiary_id": bene,
                    "specialty": spec,
                    "region": prov_meta[provider_id]["region"],
                }
                # First copy is the legitimate original; the rest are duplicates.
                records.append(
                    _finalize(rec, code, units, svc, gt_duplicate=(c > 0))
                )

    # ---- 3. Impossible-day providers (>24h billed in a single day) -----------
    for provider_id in prov_by_arch.get("impossible_day", []):
        spec = prov_meta[provider_id]["specialty"]
        n_bad_days = int(rng.integers(2, 5))
        for _ in range(n_bad_days):
            svc = _service_date()
            # Pack enough claims to blow past 1440 minutes (24h) in one day.
            minutes_total = 0
            while minutes_total < 1700:
                code = _pick_code(spec, upcode=False)
                units = int(rng.integers(1, 3))
                _, minutes = CODE_LOOKUP[(spec, code)]
                minutes_total += minutes * units
                rec = _base_record(provider_id)
                records.append(
                    _finalize(rec, code, units, svc, gt_impossible_day=True)
                )

    # ---- 4. Excessive-units claims -------------------------------------------
    for provider_id in prov_by_arch.get("unit_abuser", []):
        spec = prov_meta[provider_id]["specialty"]
        n_bad = int(rng.integers(10, 25))
        for _ in range(n_bad):
            code = _pick_code(spec, upcode=False)
            units = int(rng.integers(40, 100))  # wildly excessive units
            rec = _base_record(provider_id)
            records.append(
                _finalize(rec, code, units, _service_date(), gt_unit_outlier=True)
            )

    # ---- 5. Beneficiary-volume abusers (few benes, huge claim counts) --------
    for provider_id in prov_by_arch.get("volume_abuser", []):
        spec = prov_meta[provider_id]["specialty"]
        # Funnel a large number of claims through a tiny pool of beneficiaries.
        small_pool = rng.choice(beneficiaries, size=int(rng.integers(3, 6)), replace=False)
        n_bad = int(rng.integers(150, 300))
        for _ in range(n_bad):
            code = _pick_code(spec, upcode=False)
            units = int(rng.integers(1, 3))
            rec = {
                "claim_id": _next_claim_id(),
                "provider_id": provider_id,
                "beneficiary_id": str(rng.choice(small_pool)),
                "specialty": spec,
                "region": prov_meta[provider_id]["region"],
            }
            records.append(_finalize(rec, code, units, _service_date()))

    claims = pd.DataFrame.from_records(records)

    # Attach the provider archetype (ground truth, provider level) for tests.
    claims = claims.merge(
        providers[["provider_id", "archetype"]], on="provider_id", how="left"
    )

    # Stable ordering and a clean index.
    claims = claims.sort_values("service_date").reset_index(drop=True)
    return claims


def provider_ground_truth(claims: pd.DataFrame) -> pd.DataFrame:
    """Return the provider-level ground-truth archetype table.

    Business question: which providers are the known bad actors (and of what
    type) so detection can be scored against truth?
    """
    return (
        claims[["provider_id", "specialty", "region", "archetype"]]
        .drop_duplicates("provider_id")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    df = generate_claims()
    print(f"Generated {len(df):,} claims across {df['provider_id'].nunique():,} providers")
    print(f"Beneficiaries: {df['beneficiary_id'].nunique():,}")
    print(f"Total billed: ${df['billed_amount'].sum():,.0f}")
    print(df["archetype"].value_counts())
    print(df.head())
