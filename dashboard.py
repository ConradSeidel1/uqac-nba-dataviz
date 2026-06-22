"""Dashboard de visualisations — affiché au-dessus du chat dans app.py.

Chaque visualisation répond à une QUESTION MÉTIER du projet et possède SES PROPRES
filtres (pas de filtre global). Source : data/clean/bridge.parquet.

  V1 — Scatter salaire × performance        [saison, métrique]  → Q1/Q5 rentables & bargains
  V2 — Masse salariale × victoires (équipe)  [saison]            → Q2 budget vs résultats
  V3 — « Durée de contrat » (ancienneté) × perf [saison]         → Q4 effet ancienneté
  V4 — Valeur (perf/M$) selon l'âge          [métrique]          → Q6 valeur vs âge

Note Q3 (postes) : hors périmètre, le poste n'est pas dans les données nba_api extraites.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BRIDGE_PATH = Path(__file__).resolve().parent / "data" / "clean" / "bridge.parquet"

METRICS = {"pts": "Points", "reb": "Rebonds", "ast": "Passes",
           "stl": "Interceptions", "blk": "Contres"}


def _start_year(season: str) -> int:
    return int(str(season).split("-")[0])


@st.cache_data(show_spinner=False)
def load_bridge() -> pd.DataFrame:
    df = pd.read_parquet(BRIDGE_PATH)
    df["salary_m"] = df["salary"] / 1_000_000
    return df


# --------------------------------------------------------------------------- #
# V1 — Q1/Q5 : scatter salaire × performance
# --------------------------------------------------------------------------- #

def fig_scatter(df: pd.DataFrame, season: str, metric: str):
    sub = df[(df.season == season) & (df.salary_m > 0)]
    fig = px.scatter(
        sub, x="salary_m", y=metric, hover_name="player_name",
        color="team_abbreviation", size="gp",
        labels={"salary_m": "Salaire (M$)", metric: METRICS[metric]},
        title=f"Salaire × {METRICS[metric]} — {season} "
              f"(en haut à gauche = bons rapports / bargains)")
    fig.update_layout(showlegend=False, height=430)
    return fig


# --------------------------------------------------------------------------- #
# V2 — Q2 : masse salariale × victoires (par équipe)
# --------------------------------------------------------------------------- #

def fig_payroll_wins(df: pd.DataFrame, season: str):
    sub = df[df.season == season]
    g = (sub.groupby("team_abbreviation")
            .agg(payroll_m=("salary_m", "sum"), wins=("w", "max"))
            .reset_index())
    fig = px.scatter(
        g, x="payroll_m", y="wins", text="team_abbreviation",
        labels={"payroll_m": "Masse salariale (M$)", "wins": "Victoires"},
        title=f"Masse salariale × victoires par équipe — {season}",
        trendline="ols")
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(height=430)
    return fig


# --------------------------------------------------------------------------- #
# V3 — Q4 : « durée de contrat » (ancienneté) × performance
# --------------------------------------------------------------------------- #

def fig_tenure(df: pd.DataFrame, season: str, metric: str):
    sub = df[(df.season == season) & (df.gp >= 20)]
    g = (sub.groupby("team_tenure")
            .agg(perf=(metric, "mean"), salary_m=("salary_m", "mean"),
                 players=("player_id", "nunique"))
            .reset_index())
    fig = px.bar(
        g, x="team_tenure", y="perf",
        labels={"team_tenure": "Années dans l'équipe (durée de contrat)",
                "perf": f"{METRICS[metric]} moyens"},
        title=f"Performance moyenne selon l'ancienneté dans l'équipe — {season}",
        color="salary_m", color_continuous_scale="Viridis",
        hover_data={"players": True, "salary_m": ":.1f"})
    fig.update_layout(height=430, coloraxis_colorbar_title="Salaire moy. (M$)")
    return fig


# --------------------------------------------------------------------------- #
# V4 — Q6 : valeur (perf/M$) selon l'âge
# --------------------------------------------------------------------------- #

def fig_value_age(df: pd.DataFrame, metric: str):
    sub = df[(df.salary_m > 0) & (df.gp >= 20) & df.age.notna()].copy()
    sub["value"] = sub[metric] / sub["salary_m"]
    g = (sub.groupby("age")
            .agg(value=("value", "mean"), salary_m=("salary_m", "mean"),
                 perf=(metric, "mean"), players=("player_id", "count"))
            .reset_index())
    g = g[g.players >= 5]  # âges trop rares = bruit
    fig = px.line(
        g, x="age", y="value", markers=True,
        labels={"age": "Âge", "value": f"{METRICS[metric]} par M$"},
        title=f"Rentabilité ({METRICS[metric]}/M$) selon l'âge "
              f"— toutes saisons (le pic = meilleur rapport)")
    fig.update_layout(height=430)
    return fig


# --------------------------------------------------------------------------- #
# Rendu : chaque viz a SES filtres, dans son propre bloc
# --------------------------------------------------------------------------- #

def render_dashboard() -> None:
    try:
        df = load_bridge()
    except FileNotFoundError:
        st.warning("Données indisponibles : lance `python build_clean_datasets.py`.")
        return

    seasons = sorted(df.season.unique(), key=_start_year)
    last = len(seasons) - 1

    st.subheader("📊 Tableau de bord")

    # --- Ligne 1 : V1 (Q1/Q5) et V2 (Q2) ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rentabilité des joueurs** — _Q1 / Q5_")
        f1a, f1b = st.columns(2)
        s1 = f1a.selectbox("Saison", seasons, index=last, key="v1_season")
        m1 = f1b.selectbox("Métrique", list(METRICS),
                           format_func=lambda m: METRICS[m], key="v1_metric")
        st.plotly_chart(fig_scatter(df, s1, m1), use_container_width=True)
    with c2:
        st.markdown("**Budget vs résultats** — _Q2_")
        s2 = st.selectbox("Saison", seasons, index=last, key="v2_season")
        st.plotly_chart(fig_payroll_wins(df, s2), use_container_width=True)

    # --- Ligne 2 : V3 (Q4) et V4 (Q6) ---
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Effet « durée de contrat »** — _Q4_")
        f3a, f3b = st.columns(2)
        s3 = f3a.selectbox("Saison", seasons, index=last, key="v3_season")
        m3 = f3b.selectbox("Métrique", list(METRICS),
                           format_func=lambda m: METRICS[m], key="v3_metric")
        st.plotly_chart(fig_tenure(df, s3, m3), use_container_width=True)
    with c4:
        st.markdown("**Valeur selon l'âge** — _Q6_")
        m4 = st.selectbox("Métrique", list(METRICS),
                          format_func=lambda m: METRICS[m], key="v4_metric")
        st.plotly_chart(fig_value_age(df, m4), use_container_width=True)

    st.divider()
