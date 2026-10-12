# Program Integrity Analytics

**A Medicare/Medicaid fraud, waste & abuse (FWA) detection and investigator-prioritization dashboard.**

This project demonstrates an end-to-end healthcare program-integrity workflow: understand the
problem, structure a large claims dataset, apply the right analytic techniques to surface anomalies,
and turn the results into a **prioritized, dollar-ranked worklist** that tells investigators who to
review first and *why*.

> **Synthetic data only.** Every claim, provider, and beneficiary in this project is fabricated and
> seeded for reproducibility. There is **no real patient data, no PII, and no real provider IDs**.
> FWA signals are deliberately injected so the detection logic can be measured against ground truth.
> This is a realistic *demonstration* project — **no real-world validation is claimed.**

---

## What it does

Starting from ~70,000 synthetic claim lines (800 providers, 15,000 beneficiaries, 6 specialties,
24 months), the analytics detect five FWA patterns, each measured against a peer baseline or a hard
operational limit:

| Signal | What it catches | Method |
|---|---|---|
| **Upcoding** | Billing the specialty's most expensive codes abnormally often | Peer z-score of high-cost-code share within specialty |
| **Duplicates** | The same service billed more than once | Exact match on (provider, beneficiary, code, date) |
| **Impossible days** | More than 24h of billed service in one day | Sum of `duration x units` per provider-day vs. 1,440 min |
| **Excessive units** | Absurd unit counts on a single claim | Per-code z-score of units billed |
| **Beneficiary volume** | Phantom-patient / volume abuse | Peer z-score of claims-per-beneficiary |

These feed a composite **0–100 provider risk score** (documented weighting) and an **investigator
worklist** ranked by **risk x dollars at risk**, so finite investigative capacity is pointed at the
highest-value leads first.

## Dashboard views

1. **Executive summary** — KPIs and where the dollars at risk are concentrated.
2. **Investigator worklist** — the ranked provider table (risk score, $ at risk, "why flagged"), filterable and exportable.
3. **Signal deep-dives** — one view per FWA pattern with a chart that makes the anomaly obvious.
4. **Methodology & recommendations** — plain-English approach, assumptions, limitations, and recommended actions.

Charts follow Cole Nussbaumer Knaflic's *Storytelling with Data*: takeaway-style titles, decluttered
layouts, direct labels, and restrained color (color is reserved to highlight risk).

## Project structure

```
data_gen.py        Synthetic claims generator with injected, labeled FWA signals
analytics.py       Framework-free detection, scoring, recovery & worklist logic (pure pandas/numpy)
test_analytics.py  pytest suite (detection recall/precision, ranking, recovery math, edge cases)
app.py             Streamlit multi-view dashboard (generates data on launch)
requirements.txt   Pinned dependencies
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py          # launch the dashboard
python -m pytest -q           # run the test suite
```

Data is generated in-memory on launch, so **no data files or secrets are required** — the app is
deploy-ready to Streamlit Community Cloud as-is (main file: `app.py`).

## Testing

The suite validates that each detector recovers the injected ground-truth signals (recall/precision
sanity), that the composite risk score ranks known-bad providers above clean ones, that the recovery
math is arithmetically correct, and that nothing crashes on empty or single-provider input.

---

*Built by Leo Chan as a portfolio demonstration of healthcare program-integrity analytics.*
