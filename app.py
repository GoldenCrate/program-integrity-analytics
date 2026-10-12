"""
Program Integrity Analytics — Streamlit dashboard.

A Medicare/Medicaid fraud-waste-abuse (FWA) detection and investigator-
prioritization tool built on fully synthetic claims data. Data is generated on
launch (no data files or secrets required), so the app is deploy-ready as-is.

Visual design follows Cole Nussbaumer Knaflic's "Storytelling with Data":
takeaway-style titles (the insight, not just the metric), decluttered charts,
direct labels, and restrained color — color is reserved to highlight risk,
everything else stays neutral gray.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as a
import data_gen

# ---------------------------------------------------------------------------
# Restrained palette: gray = normal/context, red = risk/highlight.
# ---------------------------------------------------------------------------
GRAY = "#B8BEC8"
GRAY_DARK = "#6B7280"
RISK = "#C0392B"
RISK_SOFT = "#E8A5A0"
INK = "#1F2933"

st.set_page_config(
    page_title="Program Integrity Analytics",
    page_icon="shield",
    layout="wide",
)


@st.cache_data(show_spinner="Generating synthetic claims data...")
def load_data():
    claims = data_gen.generate_claims()
    risk = a.provider_risk_score(claims)
    worklist = a.investigator_worklist(claims)
    recovery = a.recovery_opportunity(claims)
    kpis = a.summary_kpis(claims)
    return claims, risk, worklist, recovery, kpis


claims, risk, worklist, recovery, kpis = load_data()


def _style_fig(fig: go.Figure, title: str) -> go.Figure:
    """Apply a decluttered, storytelling-friendly layout."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=INK)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=INK, size=13),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F4", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Program Integrity Analytics")
st.sidebar.caption("Medicare / Medicaid FWA detection — synthetic demo data")
view = st.sidebar.radio(
    "View",
    [
        "Executive summary",
        "Investigator worklist",
        "Signal deep-dives",
        "Methodology & recommendations",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"{kpis['n_claims']:,} synthetic claims · {kpis['n_providers']:,} providers\n\n"
    "No real patient data, PII, or provider IDs."
)


def fmt_money(x: float) -> str:
    if x >= 1e6:
        return f"${x/1e6:,.1f}M"
    if x >= 1e3:
        return f"${x/1e3:,.0f}K"
    return f"${x:,.0f}"


# ===========================================================================
# VIEW 1 — Executive summary
# ===========================================================================
if view == "Executive summary":
    st.title("Where the money at risk is concentrated")
    st.markdown(
        f"**So what:** of **{fmt_money(kpis['total_billed'])}** billed across "
        f"{kpis['n_providers']:,} providers, an estimated **{fmt_money(kpis['dollars_at_risk'])}** "
        f"({kpis['pct_at_risk']:.1f}%) is recoverable — and it is concentrated in a small number of providers. "
        "Working the top of the list first captures most of the opportunity."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total billed", fmt_money(kpis["total_billed"]))
    c2.metric("Dollars at risk", fmt_money(kpis["dollars_at_risk"]),
              f"{kpis['pct_at_risk']:.1f}% of billed")
    c3.metric("High-risk providers", f"{kpis['n_high_risk_providers']:,}",
              f"of {kpis['n_providers']:,}")
    c4.metric("Claims touched by a signal", f"{kpis['pct_claims_flagged']:.1f}%")

    st.markdown("---")

    left, right = st.columns([3, 2])

    with left:
        # Concentration: cumulative share of dollars at risk by provider rank.
        rec_sorted = recovery.sort_values("total_recovery", ascending=False).reset_index(drop=True)
        rec_sorted["cum_share"] = (
            rec_sorted["total_recovery"].cumsum() / rec_sorted["total_recovery"].sum() * 100
        )
        rec_sorted["rank"] = rec_sorted.index + 1
        n20 = max(1, int(np.ceil(0.2 * len(rec_sorted))))
        share_top20 = rec_sorted.loc[n20 - 1, "cum_share"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rec_sorted["rank"], y=rec_sorted["cum_share"],
            mode="lines", line=dict(color=GRAY_DARK, width=2.5), fill="tozeroy",
            fillcolor="rgba(184,190,200,0.25)",
        ))
        fig.add_vline(x=n20, line=dict(color=RISK, width=1.5, dash="dash"))
        fig.add_annotation(
            x=n20, y=share_top20,
            text=f"Top 20% of flagged providers<br>= {share_top20:.0f}% of $ at risk",
            showarrow=True, arrowhead=0, ax=70, ay=-30,
            font=dict(color=RISK, size=12), bgcolor="white",
        )
        fig = _style_fig(fig, "A few providers drive most of the dollars at risk")
        fig.update_xaxes(title="Flagged providers, ranked by $ at risk")
        fig.update_yaxes(title="Cumulative % of $ at risk", range=[0, 101])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        # Dollars at risk by FWA pattern.
        pattern_totals = {
            "Upcoding": recovery["recovery_upcoding"].sum(),
            "Duplicates": recovery["recovery_duplicate"].sum(),
            "Impossible days": recovery["recovery_impossible"].sum(),
            "Excessive units": recovery["recovery_units"].sum(),
        }
        pt = pd.Series(pattern_totals).sort_values()
        colors = [RISK if v == pt.max() else GRAY for v in pt.values]
        fig = go.Figure(go.Bar(
            x=pt.values, y=pt.index, orientation="h",
            marker_color=colors,
            text=[fmt_money(v) for v in pt.values], textposition="outside",
        ))
        fig = _style_fig(fig, "Excessive units & impossible days lead the exposure")
        fig.update_xaxes(title="Estimated $ at risk", showticklabels=False,
                         range=[0, pt.max() * 1.25])
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Reading note: dollars at risk is a conservative estimate of recoverable overbilling, "
        "not confirmed fraud. Flags are investigative leads that warrant review."
    )


# ===========================================================================
# VIEW 2 — Investigator worklist
# ===========================================================================
elif view == "Investigator worklist":
    st.title("Who to investigate first — ranked by risk x dollars")
    st.markdown(
        "**So what:** this is the operational deliverable. Each provider is ranked by a priority "
        "score (risk score x dollars at risk) so a limited investigative team recovers the most "
        "money from the riskiest actors first. Every row states *why* it was flagged."
    )

    f1, f2, f3 = st.columns([2, 2, 2])
    specialties = ["All"] + sorted(worklist["specialty"].unique())
    sel_spec = f1.selectbox("Specialty", specialties)
    min_risk = f2.slider("Minimum risk score", 0, 100, 0, 5)
    min_dollars = f3.number_input("Minimum $ at risk", 0, int(worklist["dollars_at_risk"].max()),
                                  0, step=50_000)

    wl = worklist.copy()
    if sel_spec != "All":
        wl = wl[wl["specialty"] == sel_spec]
    wl = wl[(wl["risk_score"] >= min_risk) & (wl["dollars_at_risk"] >= min_dollars)]

    st.markdown(
        f"**{len(wl):,}** providers match — totaling "
        f"**{fmt_money(wl['dollars_at_risk'].sum())}** in dollars at risk."
    )

    display = wl.copy()
    display["dollars_at_risk"] = display["dollars_at_risk"].map(fmt_money)
    display = display.rename(columns={
        "priority_rank": "Rank", "provider_id": "Provider", "specialty": "Specialty",
        "risk_score": "Risk score", "dollars_at_risk": "$ at risk", "reasons": "Why flagged",
    })[["Rank", "Provider", "Specialty", "Risk score", "$ at risk", "Why flagged"]]

    st.dataframe(
        display, use_container_width=True, hide_index=True, height=520,
        column_config={
            "Risk score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=float(worklist["risk_score"].max()),
                format="%.1f",
            ),
        },
    )

    st.download_button(
        "Download worklist (CSV)",
        wl.to_csv(index=False).encode(),
        file_name="investigator_worklist.csv",
        mime="text/csv",
    )


# ===========================================================================
# VIEW 3 — Signal deep-dives
# ===========================================================================
elif view == "Signal deep-dives":
    st.title("What the anomaly looks like, signal by signal")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Upcoding", "Duplicates", "Impossible days", "Excessive units"]
    )

    # ---- Upcoding ----
    with tab1:
        upc = a.detect_upcoding(claims)
        st.markdown(
            "**So what:** upcoders bill their specialty's most expensive codes far more often than "
            "peers. Each dot is a provider; the red dots sit well outside the normal band."
        )
        upc = upc.copy()
        upc["flag_label"] = np.where(upc["flagged"], "Flagged", "Within peer norm")
        fig = px.strip(
            upc, x="specialty", y="high_cost_share", color="flag_label",
            color_discrete_map={"Flagged": RISK, "Within peer norm": GRAY},
        )
        fig.update_traces(jitter=0.35, marker=dict(size=6, opacity=0.6))
        fig = _style_fig(fig, "Flagged providers bill high-cost codes far above peers")
        fig.update_layout(showlegend=True, legend=dict(title=""))
        fig.update_yaxes(title="Share of claims on high-cost codes", tickformat=".0%")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Duplicates ----
    with tab2:
        dup = a.detect_duplicates(claims)
        st.markdown(
            "**So what:** the same service (same provider, beneficiary, code, and date) billed "
            "more than once. These providers have the most duplicate claim lines."
        )
        top_dup = (
            dup.groupby("provider_id")["claim_id"].count()
            .sort_values(ascending=False).head(15).iloc[::-1]
        )
        colors = [RISK if v == top_dup.max() else RISK_SOFT for v in top_dup.values]
        fig = go.Figure(go.Bar(
            x=top_dup.values, y=top_dup.index, orientation="h",
            marker_color=colors, text=top_dup.values, textposition="outside",
        ))
        fig = _style_fig(fig, "Top providers by number of duplicate claim lines")
        fig.update_xaxes(title="Duplicate claim lines", showticklabels=False,
                         range=[0, top_dup.max() * 1.2])
        fig.update_yaxes(title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(dup):,} duplicate claim lines across {dup['provider_id'].nunique()} providers.")

    # ---- Impossible days ----
    with tab3:
        imp = a.detect_impossible_days(claims)
        st.markdown(
            "**So what:** no provider can deliver more than 24 hours of service in a day. "
            "Every point above the red 24-hour line is a physically impossible billed day."
        )
        sample = imp.sort_values("billed_hours", ascending=False).head(60)
        fig = go.Figure(go.Scatter(
            x=list(range(len(sample))), y=sample["billed_hours"],
            mode="markers", marker=dict(size=9, color=RISK, opacity=0.7),
        ))
        fig.add_hline(y=24, line=dict(color=INK, width=1.5, dash="dash"))
        fig.add_annotation(x=len(sample) * 0.5, y=24, yshift=12,
                           text="24-hour ceiling", showarrow=False,
                           font=dict(color=INK, size=12))
        fig = _style_fig(fig, "Flagged provider-days bill well beyond 24 hours")
        fig.update_yaxes(title="Billed hours in a single day")
        fig.update_xaxes(title="Flagged provider-days (top 60)", showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(imp):,} impossible provider-days across {imp['provider_id'].nunique()} providers.")

    # ---- Excessive units ----
    with tab4:
        uo = a.detect_unit_outliers(claims)
        st.markdown(
            "**So what:** a handful of claims bill absurd quantities of a single procedure. "
            "The red points are flagged outliers against the normal gray distribution of units."
        )
        normal = claims[~claims["claim_id"].isin(set(uo["claim_id"]))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=normal["units"].sample(min(4000, len(normal)), random_state=1),
            y=np.random.default_rng(1).normal(0, 1, min(4000, len(normal))),
            mode="markers", marker=dict(size=4, color=GRAY, opacity=0.35),
        ))
        fig.add_trace(go.Scatter(
            x=uo["units"], y=np.random.default_rng(2).normal(0, 1, len(uo)),
            mode="markers", marker=dict(size=7, color=RISK, opacity=0.7),
        ))
        fig = _style_fig(fig, "Flagged claims bill far more units than normal")
        fig.update_xaxes(title="Units billed on a single claim")
        fig.update_yaxes(title="", showticklabels=False, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(uo):,} excessive-unit claims across {uo['provider_id'].nunique()} providers.")


# ===========================================================================
# VIEW 4 — Methodology & recommendations
# ===========================================================================
elif view == "Methodology & recommendations":
    st.title("Methodology, assumptions & recommended actions")

    st.header("Approach")
    st.markdown(
        """
This project demonstrates an end-to-end program-integrity workflow of the kind Emendata runs
for Medicare/Medicaid clients: **understand the problem, structure a large claims dataset,
apply the right analytic technique, and turn results into prioritized action.**

1. **Structure the data.** ~70,000 synthetic claim lines across 800 providers, 15,000
   beneficiaries, 6 specialties, and 24 months — each line carrying cost, units, procedure
   duration, and service date.
2. **Detect anomalies against a peer baseline.** Fraud is rarely provable from one claim, so
   each detector measures a provider against *peers in the same specialty* or against a hard
   operational limit:
   - **Upcoding** — share of claims on the specialty's highest-cost codes, z-scored vs. peers.
   - **Duplicates** — identical (provider, beneficiary, code, date) claim lines billed more than once.
   - **Impossible days** — more than 24 hours of billed service time in one calendar day.
   - **Excessive units** — unit counts z-scored within each procedure code.
   - **Beneficiary volume** — claims-per-beneficiary z-scored vs. peers (phantom-patient pattern).
3. **Score and prioritize.** A composite 0-100 risk score (documented weighting) combines the
   signals, and the investigator worklist ranks providers by **risk x dollars at risk** so finite
   investigative capacity is pointed at the highest-value leads.
        """
    )

    st.header("Risk-score weighting")
    wdf = pd.DataFrame(
        {"Signal": list(a.RISK_WEIGHTS.keys()), "Weight": list(a.RISK_WEIGHTS.values())}
    )
    wdf["Signal"] = wdf["Signal"].map({
        "upcoding": "Upcoding", "duplicate": "Duplicates", "impossible": "Impossible days",
        "units": "Excessive units", "volume": "Beneficiary volume",
    })
    wdf["Weight"] = (wdf["Weight"] * 100).map(lambda x: f"{x:.0f}%")
    st.table(wdf)
    st.caption(
        "Upcoding and impossible days carry the most weight: the first is the clearest intentional "
        "overbilling pattern, the second is a hard operational impossibility."
    )

    st.header("Assumptions & limitations")
    st.markdown(
        """
- **Synthetic data only.** All claims, providers, and beneficiaries are fabricated and seeded for
  reproducibility. There is no real PHI/PII, and **no real-world validation is claimed**. FWA
  signals were deliberately injected so the detection logic is measurable against ground truth.
- **Flags are leads, not verdicts.** A high score means "review first," not "confirmed fraud."
  Peer-relative methods can surface legitimate high-complexity practices; human adjudication stays
  in the loop.
- **Recovery is a conservative estimate** of recoverable overbilling (e.g., duplicate copies at full
  value, upcoding as excess over the specialty median), not a guaranteed collection figure.
- **Single-specialty peer groups** assume specialty is the right comparison cohort; in production
  you would also risk-adjust for case mix, geography, and patient acuity.
        """
    )

    st.header("Recommended actions")
    st.markdown(
        """
1. **Work the top of the worklist first.** The dollars at risk are concentrated in a small set of
   providers — prioritizing them captures most of the recovery opportunity for the least effort.
2. **Fast-track the impossible-day and duplicate providers.** These are the least ambiguous signals
   (a physical impossibility and an exact-match rebill) and make the strongest, quickest cases.
3. **Route upcoding and volume flags to clinical review.** These need case-mix context; pair the
   analytics with a clinician to confirm before any recovery action.
4. **Operationalize as a monitor, not a one-off.** Re-run monthly, track score movement per provider,
   and feed confirmed outcomes back to re-tune detector thresholds and weights over time.
        """
    )

    st.markdown("---")
    st.caption(
        "Built by Leo Chan as a portfolio demonstration. Synthetic data, framework-free analytics "
        "(pandas/numpy), tested with pytest, visualized with Plotly + Streamlit."
    )
