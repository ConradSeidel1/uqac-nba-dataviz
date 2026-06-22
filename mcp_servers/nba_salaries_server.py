"""Serveur MCP « nba-salaries » — salaires (source : dataset Kaggle nettoyé).

Lit data/clean/salaries_clean.parquet et data/clean/bridge.parquet (lien stats↔salaire).
Aucun appel réseau : tout est lu en local.

Outils : list_seasons, get_player_salary, get_team_payroll, query_salaries,
get_value_ranking.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

DATA = Path(__file__).resolve().parent.parent / "data" / "clean"
SALARIES_PATH = DATA / "salaries_clean.parquet"
BRIDGE_PATH = DATA / "bridge.parquet"

_sal: pd.DataFrame | None = None
_bridge: pd.DataFrame | None = None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


def salaries() -> pd.DataFrame:
    global _sal
    if _sal is None:
        if not SALARIES_PATH.exists():
            raise FileNotFoundError(
                f"{SALARIES_PATH} introuvable. Lance d'abord build_clean_datasets.py.")
        _sal = pd.read_parquet(SALARIES_PATH)
        if "name_norm" not in _sal.columns:
            _sal["name_norm"] = _sal["player_name"].map(_norm)
    return _sal


def bridge() -> pd.DataFrame:
    global _bridge
    if _bridge is None:
        if not BRIDGE_PATH.exists():
            raise FileNotFoundError(
                f"{BRIDGE_PATH} introuvable. Lance d'abord build_clean_datasets.py.")
        _bridge = pd.read_parquet(BRIDGE_PATH)
    return _bridge


def _start_year(season: str) -> int:
    return int(str(season).split("-")[0])


def as_table(df: pd.DataFrame, total_players: int | None = None,
             players_returned: int | None = None) -> dict:
    """Format compact {columns, rows, n_rows, ...} + métadonnées de troncature."""
    cols = list(df.columns)
    rows = df.where(pd.notna(df), None).values.tolist()
    out = {"columns": cols, "rows": rows, "n_rows": len(rows)}
    if total_players is not None:
        out["players_returned"] = players_returned
        out["total_players"] = total_players
        out["truncated"] = (players_returned or 0) < total_players
    return out


def top_players_detailed(df: pd.DataFrame, rank_col: str, limit: int,
                         ascending: bool) -> tuple[pd.DataFrame, int, int]:
    """Top `limit` JOUEURS (classés sur la moyenne de `rank_col`), TOUTES leurs saisons."""
    order = (df.groupby("player_name")[rank_col].mean()
               .sort_values(ascending=ascending))
    total = len(order)
    keep = list(order.head(limit).index)
    sub = df[df.player_name.isin(keep)].copy()
    sub["_rank"] = sub.player_name.map({p: i for i, p in enumerate(keep)})
    sub = sub.sort_values(["_rank", "season"]).drop(columns="_rank")
    return sub, len(keep), total


def filter_seasons(df: pd.DataFrame, season_from: str | None,
                   season_to: str | None) -> pd.DataFrame:
    """Restreint à une plage de saisons [season_from, season_to] incluses (optionnelles)."""
    out = df
    if season_from:
        out = out[out["season"].map(_start_year) >= _start_year(season_from)]
    if season_to:
        out = out[out["season"].map(_start_year) <= _start_year(season_to)]
    return out


mcp = FastMCP("nba-salaries")


@mcp.tool()
def list_seasons() -> dict:
    """Saisons couvertes par les salaires + bornes de la plage totale."""
    seasons = sorted(salaries().season.unique(), key=_start_year)
    return {"seasons": seasons,
            "season_from": seasons[0] if seasons else None,
            "season_to": seasons[-1] if seasons else None,
            "count": len(seasons)}


@mcp.tool()
def get_player_salary(player: str, season_from: str | None = None,
                      season_to: str | None = None) -> dict:
    """Salaire d'un joueur sur une PLAGE de saisons (1 ligne par saison, tableau compact).

    `season_from` / `season_to` (ex. '2018-19') bornent la période ; défaut = tout.
    Salaire nominal et ajusté à l'inflation.
    """
    sub = salaries()
    sub = sub[sub.name_norm.str.contains(_norm(player), regex=False)]
    sub = filter_seasons(sub, season_from, season_to)
    cols = [c for c in ["player_name", "season", "salary", "inflation_adj_salary"]
            if c in sub.columns]
    out = sub[cols].sort_values("season", key=lambda s: s.map(_start_year))
    return as_table(out)


@mcp.tool()
def get_team_payroll(team: str, season: str) -> dict:
    """Masse salariale d'une équipe pour une saison (ex. team='LAL', season='2018-19')."""
    b = bridge()
    if "team_abbreviation" not in b.columns:
        return {"error": "team_abbreviation absent du pont ; régénère build_clean_datasets.py"}
    sub = b[(b.season == season) & (b.team_abbreviation.str.upper() == team.upper())]
    if sub.empty:
        return {"team": team, "season": season, "total_salary": 0, "players": 0}
    return {"team": team.upper(), "season": season,
            "total_salary": int(sub.salary.fillna(0).sum()),
            "players": int(sub.salary.notna().sum())}


@mcp.tool()
def query_salaries(season_from: str | None = None, season_to: str | None = None,
                   min_salary: int = 0, aggregate: bool = False,
                   limit: int = 10) -> dict:
    """Salaires sur une PLAGE de saisons (tableau compact). Mieux payé en tête.

    - `season_from` / `season_to` (ex. '2018-19') : plage, incluses ; défaut = tout.
    - `min_salary` filtre les salaires inférieurs au seuil.
    - `aggregate=False` (défaut) : `limit` = nombre de JOUEURS, et pour chacun TOUTES
      ses saisons sont incluses (1 ligne par joueur-saison) — idéal pour visualiser.
    - `aggregate=True` : résumé par joueur (salaire moyen + total + nb de saisons).
    - `limit` = nombre de joueurs (défaut 10).
    """
    sub = filter_seasons(salaries(), season_from, season_to)
    sub = sub[sub.salary.fillna(0) >= min_salary]
    if sub.empty:
        return {"error": "aucun salaire pour cette plage / ce filtre"}

    if aggregate:
        agg = (sub.groupby("name_norm", as_index=False)
                  .agg(player_name=("player_name", "first"),
                       salary_avg=("salary", "mean"), salary_total=("salary", "sum"),
                       seasons=("season", "nunique")))
        total = len(agg)
        agg = agg.sort_values("salary_avg", ascending=False).head(limit)
        agg["salary_avg"] = agg.salary_avg.round(0).astype("Int64")
        agg["salary_total"] = agg.salary_total.astype("Int64")
        return as_table(agg[["player_name", "salary_avg", "salary_total", "seasons"]],
                        total_players=total, players_returned=len(agg))

    sub2, n_kept, total = top_players_detailed(sub, "salary", limit, ascending=False)
    cols = [c for c in ["player_name", "season", "salary", "inflation_adj_salary"]
            if c in sub2.columns]
    return as_table(sub2[cols], total_players=total, players_returned=n_kept)


@mcp.tool()
def get_value_ranking(metric: str = "pts", season_from: str | None = None,
                      season_to: str | None = None, min_games: int = 20,
                      order: str = "best", aggregate: bool = False,
                      limit: int = 10) -> dict:
    """Rentabilité : performance rapportée au salaire, sur une PLAGE (tableau compact).

    value = metric / (salary / 1_000_000). Trié : 1re ligne = plus rentable (order='best')
    ou plus surpayé (order='worst').
    - `metric` ∈ {pts, reb, ast, stl, blk}.
    - `season_from` / `season_to` (ex. '2018-19') : plage, incluses ; défaut = tout.
    - `min_games` écarte les saisons trop peu jouées.
    - `order` = "best" (bargains) ou "worst" (surpayés).
    - `aggregate=False` (défaut) : `limit` = nombre de JOUEURS, et pour chacun TOUTES
      ses saisons sont incluses (1 ligne par joueur-saison) — idéal pour visualiser.
    - `aggregate=True` : résumé par joueur (moyennes sur la période).
    - `limit` = nombre de joueurs (défaut 10).

    Outil pour « le plus rentable », « les bargains », « les surpayés ».
    """
    metric = metric.lower()
    allowed = {"pts", "reb", "ast", "stl", "blk"}
    if metric not in allowed:
        return {"error": f"metric inconnue '{metric}'. Choix : {', '.join(sorted(allowed))}"}

    b = bridge()
    needed = {metric, "gp", "salary", "player_name", "season"}
    if not needed.issubset(b.columns):
        return {"error": "colonnes manquantes dans bridge.parquet ; régénère build_clean_datasets.py"}

    sub = filter_seasons(b, season_from, season_to).copy()
    sub = sub[(sub.salary.fillna(0) > 0) & (sub.gp.fillna(0) >= min_games)]
    if sub.empty:
        return {"error": f"aucune ligne avec salaire et gp >= {min_games} sur la période"}

    ascending = (order == "worst")

    if aggregate:
        agg = (sub.groupby(["player_id", "player_name"], as_index=False)
                  .agg(metric_avg=(metric, "mean"), salary_avg=("salary", "mean"),
                       gp_total=("gp", "sum"), seasons=("season", "nunique")))
        agg["value"] = (agg.metric_avg / (agg.salary_avg / 1_000_000)).round(3)
        agg["metric_avg"] = agg.metric_avg.round(1)
        agg["salary_avg"] = agg.salary_avg.round(0).astype("Int64")
        total = len(agg)
        agg = agg.sort_values("value", ascending=ascending).head(limit)
        out = agg[["player_name", "metric_avg", "salary_avg", "value", "gp_total", "seasons"]]
        out = out.rename(columns={"metric_avg": f"{metric}_avg",
                                  "value": f"{metric}_per_million"})
        return as_table(out, total_players=total, players_returned=len(agg))

    sub["value"] = (sub[metric] / (sub.salary / 1_000_000)).round(3)
    sub2, n_kept, total = top_players_detailed(sub, "value", limit, ascending)
    out = sub2[["player_name", "season", metric, "salary", "gp", "value"]]
    out = out.rename(columns={"value": f"{metric}_per_million"})
    return as_table(out, total_players=total, players_returned=n_kept)


if __name__ == "__main__":
    mcp.run()
