"""Dashboard de visualisations — affiché au-dessus du chat dans app.py.

Quatre visualisations Plotly construites sur la base propre `data/clean/bridge.parquet`
(joueur, saison, équipe, stats pts/reb/ast/stl/blk, salaire) :

  V1. Scatter salaire × performance   (la vue centrale : bargains vs surpayés)
  V2. Top joueurs (barres)            (mieux payés OU plus rentables)
  V3. Évolution temporelle            (salaire moyen et stat moyenne par saison)
  V4. Masse salariale par équipe      (barres, pour une saison donnée)

Le dashboard expose des filtres (saison, métrique) et reste indépendant du chat :
il lit directement le Parquet, sans passer par les serveurs MCP.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BRIDGE_PATH = Path(__file__).resolve().parent / "data" / "clean" / "bridge.parquet"

# Métriques de performance proposées dans les filtres.
METRICS = {
    "pts": "Points", "reb": "Rebonds", "ast": "Passes",
    "stl": "Interceptions", "blk": "Contres",
}


def _start_year(season: str) -> int:
    return int(str(season).split("-")[0])


@st.cache_data(show_spinner=False)
def load_bridge() -> pd.DataFrame:
    """Charge (et met en cache) la base jointe stats↔salaires."""
    df = pd.read_parquet(BRIDGE_PATH)
    df["salary_m"] = df["salary"] / 1_000_000  # salaire en millions, plus lisible
    return df


# --------------------------------------------------------------------------- #
# Visualisations
# --------------------------------------------------------------------------- #

def fig_scatter(df: pd.DataFrame, season: str, metric: str):
    """V1 — Scatter salaire (X) × performance (Y) pour une saison."""
    sub = df[(df.season == season) & (df.salary_m > 0)]
    fig = px.scatter(
        sub, x="salary_m", y=metric, hover_name="player_name",
        color="team_abbreviation", size="gp",
        labels={"salary_m": "Salaire (M$)", metric: METRICS.get(metric, metric),
                "team_abbreviation": "Équipe"},
        title=f"Salaire × {METRICS.get(metric, metric)} — {season}")
    fig.update_layout(showlegend=False, height=420)
    return fig


def fig_top_players(df: pd.DataFrame, season: str, metric: str, mode: str):
    """V2 — Top 15 joueurs : mieux payés OU plus rentables (perf / M$)."""
    sub = df[(df.season == season) & (df.salary_m > 0)].copy()
    if mode == "Plus rentables (perf/M$)":
        sub["val"] = sub[metric] / sub["salary_m"]
        sub = sub[sub.gp >= 20]  # éviter les ratios extrêmes de faibles temps de jeu
        ylab = f"{METRICS.get(metric, metric)} par M$"
        title = f"Top 15 plus rentables ({METRICS.get(metric, metric)}/M$) — {season}"
    else:
        sub["val"] = sub["salary_m"]
        ylab = "Salaire (M$)"
        title = f"Top 15 mieux payés — {season}"
    sub = sub.sort_values("val", ascending=False).head(15)
    fig = px.bar(sub, x="player_name", y="val",
                 labels={"player_name": "", "val": ylab}, title=title,
                 color="val", color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False, height=420, xaxis_tickangle=-40)
    return fig


def fig_evolution(df: pd.DataFrame, metric: str):
    """V3 — Évolution par saison : salaire moyen et stat moyenne (double axe)."""
    g = (df.groupby("season")
           .agg(salary_m=("salary_m", "mean"), stat=(metric, "mean"))
           .reset_index())
    g = g.sort_values("season", key=lambda s: s.map(_start_year))
    fig = px.line(g, x="season", y="salary_m", markers=True,
                  labels={"season": "Saison", "salary_m": "Salaire moyen (M$)"},
                  title=f"Évolution du salaire moyen et des {METRICS.get(metric, metric).lower()} par saison")
    # Stat moyenne sur un 2e axe.
    fig.add_scatter(x=g.season, y=g.stat, name=METRICS.get(metric, metric),
                    yaxis="y2", mode="lines+markers", line=dict(color="orange"))
    fig.update_layout(
        height=420, xaxis_tickangle=-45,
        yaxis=dict(title="Salaire moyen (M$)"),
        yaxis2=dict(title=METRICS.get(metric, metric), overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1))
    return fig


def fig_team_payroll(df: pd.DataFrame, season: str):
    """V4 — Masse salariale par équipe pour une saison."""
    sub = df[df.season == season]
    g = (sub.groupby("team_abbreviation")
            .agg(payroll_m=("salary_m", "sum"), players=("player_id", "nunique"))
            .reset_index()
            .sort_values("payroll_m", ascending=False))
    fig = px.bar(g, x="team_abbreviation", y="payroll_m",
                 labels={"team_abbreviation": "Équipe", "payroll_m": "Masse salariale (M$)"},
                 title=f"Masse salariale par équipe — {season}",
                 color="payroll_m", color_continuous_scale="Reds")
    fig.update_layout(coloraxis_showscale=False, height=420)
    return fig


# --------------------------------------------------------------------------- #
# Rendu du dashboard
# --------------------------------------------------------------------------- #

def render_dashboard() -> None:
    """Affiche le bloc de visualisations avec ses filtres (au-dessus du chat)."""
    try:
        df = load_bridge()
    except FileNotFoundError:
        st.warning("Données indisponibles : lance d'abord `python build_clean_datasets.py`.")
        return

    seasons = sorted(df.season.unique(), key=_start_year)

    st.subheader("📊 Tableau de bord")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        season = st.selectbox("Saison", seasons, index=len(seasons) - 1)
    with c2:
        metric = st.selectbox("Métrique", list(METRICS),
                              format_func=lambda m: METRICS[m])
    with c3:
        top_mode = st.selectbox("Classement", ["Mieux payés", "Plus rentables (perf/M$)"])

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(fig_scatter(df, season, metric), use_container_width=True)
    with r1c2:
        st.plotly_chart(fig_top_players(df, season, metric, top_mode),
                        use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(fig_evolution(df, metric), use_container_width=True)
    with r2c2:
        st.plotly_chart(fig_team_payroll(df, season), use_container_width=True)

    st.divider()
