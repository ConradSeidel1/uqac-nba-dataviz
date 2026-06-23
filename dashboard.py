"""Dashboard de visualisations — affiché au-dessus du chat dans app.py.

Chaque visualisation répond à une question métier du projet. Les filtres de PÉRIODE
(saison début / saison fin) sont communs et placés dans la sidebar ; des filtres de
sélection (joueurs, équipes) et de métrique sont propres à chaque graphe.

  V1 — Scatter salaire × performance        → 1 point par joueur (moyenne sur la plage)
        + régression linéaire FIGÉE (calculée 1x sur tout le dataset) : au-dessus de la
        droite = bonne affaire, en-dessous = mauvaise.
  V2 — Masse salariale × victoires (équipe)  → moyenne par équipe sur la plage
  V3 — « Durée de contrat » (ancienneté) × perf → moyenne par niveau d'ancienneté
  V4 — Valeur (perf/M$) selon l'âge          → moyenne par âge sur la plage

Source : data/clean/bridge.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    df["salary_adj_m"] = df["inflation_adj_salary"] / 1_000_000
    return df


def in_range(df: pd.DataFrame, s_from: str, s_to: str) -> pd.DataFrame:
    y0, y1 = _start_year(s_from), _start_year(s_to)
    return df[df.season.map(_start_year).between(y0, y1)]


@st.cache_data(show_spinner=False)
def global_regression(metric: str) -> tuple[float, float]:
    """Régression linéaire FIGÉE perf ~ salaire ajusté inflation, sur TOUT le dataset.

    Calculée une seule fois par métrique (cache). On utilise le salaire AJUSTÉ à
    l'inflation pour que la droite de référence reste valable sur toutes les époques.
    Renvoie (pente, ordonnée) pour y = pente * salaire_adj_m + ordonnée.
    """
    df = load_bridge()
    d = df[(df.salary_adj_m > 0) & df[metric].notna() & (df.gp >= 20)]
    slope, intercept = np.polyfit(d.salary_adj_m, d[metric], 1)
    return float(slope), float(intercept)


# --------------------------------------------------------------------------- #
# V1 — scatter salaire × performance + régression figée
# --------------------------------------------------------------------------- #

def fig_scatter(df: pd.DataFrame, metric: str):
    # 1 point par joueur : moyennes sur la plage déjà filtrée.
    g = (df[df.salary_adj_m > 0]
         .groupby(["player_id", "player_name"], as_index=False)
         .agg(salary_adj_m=("salary_adj_m", "mean"), perf=(metric, "mean"),
              gp=("gp", "sum"), team=("team_abbreviation", "last")))
    fig = px.scatter(
        g, x="salary_adj_m", y="perf", hover_name="player_name",
        color="team", opacity=0.65,
        labels={"salary_adj_m": "Salaire ajusté inflation (M$)", "perf": METRICS[metric]},
        title=f"Salaire × {METRICS[metric]} — au-dessus de la droite = bonne affaire")
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))

    # Droite de régression figée (référence absolue).
    slope, intercept = global_regression(metric)
    xs = np.linspace(g.salary_adj_m.min(), g.salary_adj_m.max(), 50)
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept, mode="lines", name="Référence (régression)",
        line=dict(color="black", dash="dash", width=2)))
    fig.update_layout(showlegend=False, height=440)
    return fig


# --------------------------------------------------------------------------- #
# V2 — masse salariale × victoires (par équipe), moyenne sur la plage
# --------------------------------------------------------------------------- #

def fig_payroll_wins(df: pd.DataFrame):
    per_season = (df.groupby(["team_abbreviation", "season"])
                    .agg(payroll_m=("salary_m", "sum"), wins=("w", "max"))
                    .reset_index())
    g = (per_season.groupby("team_abbreviation")
                   .agg(payroll_m=("payroll_m", "mean"), wins=("wins", "mean"))
                   .reset_index())
    fig = px.scatter(
        g, x="payroll_m", y="wins", text="team_abbreviation", trendline="ols",
        labels={"payroll_m": "Masse salariale moyenne (M$)", "wins": "Victoires moyennes"},
        title="Masse salariale × victoires par équipe (moyenne sur la période)")
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(height=440)
    return fig


# --------------------------------------------------------------------------- #
# V3 — performance selon l'ancienneté, moyenne par niveau
# --------------------------------------------------------------------------- #

def fig_tenure(df: pd.DataFrame, metric: str):
    sub = df[df.gp >= 20]
    g = (sub.groupby("team_tenure")
            .agg(perf=(metric, "mean"), salary_m=("salary_m", "mean"),
                 players=("player_id", "nunique"))
            .reset_index())
    fig = px.bar(
        g, x="team_tenure", y="perf",
        labels={"team_tenure": "Années dans l'équipe (durée de contrat)",
                "perf": f"{METRICS[metric]} moyens"},
        title=f"Performance moyenne selon l'ancienneté dans l'équipe",
        color="salary_m", color_continuous_scale="Plasma",
        hover_data={"players": True, "salary_m": ":.1f"})
    fig.update_layout(height=440, coloraxis_colorbar_title="Salaire<br>moy. (M$)")
    return fig


# --------------------------------------------------------------------------- #
# V4 — rentabilité (perf/M$) selon l'âge, moyenne par âge
# --------------------------------------------------------------------------- #

def fig_value_age(df: pd.DataFrame, metric: str):
    sub = df[(df.salary_m > 0) & (df.gp >= 20) & df.age.notna()].copy()
    sub["value"] = sub[metric] / sub["salary_m"]
    g = (sub.groupby("age")
            .agg(value=("value", "mean"), players=("player_id", "count"))
            .reset_index())
    g = g[g.players >= 5]
    fig = px.line(
        g, x="age", y="value", markers=True,
        labels={"age": "Âge", "value": f"{METRICS[metric]} par M$"},
        title=f"Rentabilité ({METRICS[metric]}/M$) selon l'âge")
    fig.update_layout(height=440)
    return fig


# --------------------------------------------------------------------------- #
# Rendu
# --------------------------------------------------------------------------- #

def render_dashboard() -> None:
    try:
        df = load_bridge()
    except FileNotFoundError:
        st.warning("Données indisponibles : lance `python build_clean_datasets.py`.")
        return

    seasons = sorted(df.season.unique(), key=_start_year)

    # --- Filtres communs (sidebar) : période + métrique + sélection joueurs/équipes ---
    with st.sidebar:
        st.header("Filtres du tableau de bord")
        s_from = st.selectbox("Saison de début", seasons, index=0)
        s_to = st.selectbox("Saison de fin", seasons, index=len(seasons) - 1)
        if _start_year(s_from) > _start_year(s_to):
            s_from, s_to = s_to, s_from  # tolère l'inversion
        metric = st.selectbox("Métrique", list(METRICS),
                              format_func=lambda m: METRICS[m])
        teams = sorted(df.team_abbreviation.dropna().unique())
        sel_teams = st.multiselect("Équipes (vide = toutes)", teams)
        players = sorted(df.player_name.dropna().unique())
        sel_players = st.multiselect("Joueurs (vide = tous)", players)

    # Application des filtres communs.
    d = in_range(df, s_from, s_to)
    if sel_teams:
        d = d[d.team_abbreviation.isin(sel_teams)]
    if sel_players:
        d = d[d.player_name.isin(sel_players)]

    if d.empty:
        st.info("Aucune donnée pour ces filtres.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_scatter(d, metric), use_container_width=True)
    with c2:
        st.plotly_chart(fig_payroll_wins(d), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_tenure(d, metric), use_container_width=True)
    with c4:
        st.plotly_chart(fig_value_age(d, metric), use_container_width=True)

    st.divider()
