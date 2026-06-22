"""Serveur MCP « nba-stats » — statistiques de jeu (source : nba_api nettoyée).

Expose les statistiques joueur-saison de la base propre `data/clean/stats_clean.parquet`.
Aucun appel réseau : tout est lu en local.

Outils : list_seasons, search_players, get_player_season_stats, get_league_leaders.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

DATA = Path(__file__).resolve().parent.parent / "data" / "clean"
STATS_PATH = DATA / "stats_clean.parquet"

STAT_COLUMNS = {
    "pts": "points", "reb": "rebonds", "ast": "passes", "stl": "interceptions",
    "blk": "contres", "tov": "ballons perdus", "min": "minutes", "gp": "matchs joués",
    "fg_pct": "% au tir", "fg3_pct": "% à 3pts", "ft_pct": "% lancers francs",
    "plus_minus": "+/-", "age": "âge",
}

_df: pd.DataFrame | None = None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


def stats() -> pd.DataFrame:
    """Charge (une fois) la base de stats."""
    global _df
    if _df is None:
        if not STATS_PATH.exists():
            raise FileNotFoundError(
                f"{STATS_PATH} introuvable. Lance d'abord build_clean_datasets.py.")
        _df = pd.read_parquet(STATS_PATH)
        if "name_norm" not in _df.columns:
            _df["name_norm"] = _df["player_name"].map(_norm)
    return _df


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
    """Top `limit` JOUEURS (classés sur la moyenne de `rank_col`), TOUTES leurs saisons.

    Renvoie (df_filtré, n_joueurs_renvoyés, total_joueurs).
    """
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


mcp = FastMCP("nba-stats")


@mcp.tool()
def list_seasons() -> dict:
    """Saisons disponibles + bornes de la plage totale. À appeler en premier."""
    seasons = sorted(stats().season.unique(), key=_start_year)
    return {"seasons": seasons,
            "season_from": seasons[0] if seasons else None,
            "season_to": seasons[-1] if seasons else None,
            "count": len(seasons)}


@mcp.tool()
def search_players(query: str, limit: int = 20) -> list[dict]:
    """Recherche des joueurs par nom (insensible aux accents/casse)."""
    df = stats()
    hit = df[df.name_norm.str.contains(_norm(query), regex=False)]
    out = []
    for pid, g in hit.groupby("player_id"):
        out.append({"player_id": int(pid), "player_name": g.player_name.iloc[0],
                    "seasons": sorted(g.season.unique(), key=_start_year)})
        if len(out) >= limit:
            break
    return out


@mcp.tool()
def get_player_season_stats(player: str, season_from: str | None = None,
                            season_to: str | None = None) -> dict:
    """Stats d'un joueur sur une PLAGE de saisons (1 ligne par saison, tableau compact).

    `season_from` / `season_to` (ex. '2018-19') bornent la période ; défaut = tout.
    """
    sub = stats()
    sub = sub[sub.name_norm.str.contains(_norm(player), regex=False)]
    sub = filter_seasons(sub, season_from, season_to)
    cols = ["season", "player_id", "player_name", "team_abbreviation", "age", "gp",
            "min", "pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct",
            "ft_pct", "plus_minus"]
    cols = [c for c in cols if c in sub.columns]
    out = sub[cols].sort_values("season", key=lambda s: s.map(_start_year))
    return as_table(out)


@mcp.tool()
def get_league_leaders(stat: str, season_from: str | None = None,
                       season_to: str | None = None, min_games: int = 0,
                       aggregate: bool = False, limit: int = 10) -> dict:
    """Meilleurs joueurs pour une statistique sur une PLAGE de saisons (tableau compact).

    Trié décroissant : le meilleur joueur est en tête.
    `stat` ∈ {pts, reb, ast, stl, blk, tov, min, gp, fg_pct, fg3_pct, ft_pct,
    plus_minus, age}.
    - `season_from` / `season_to` (ex. '2018-19') : plage, incluses ; défaut = tout.
    - `min_games` écarte les saisons où le joueur a trop peu joué.
    - `aggregate=False` (défaut) : `limit` = nombre de JOUEURS retournés, et pour chaque
      joueur TOUTES ses saisons de la plage sont incluses (1 ligne par joueur-saison) —
      idéal pour visualiser. `total_players` / `truncated` indiquent s'il en reste.
    - `aggregate=True` : un résumé (1 ligne par joueur, moyenne sur la période).
    - `limit` = nombre de joueurs (défaut 10).
    """
    stat = stat.lower()
    if stat not in STAT_COLUMNS:
        return {"error": f"stat inconnue '{stat}'. Choix : {', '.join(STAT_COLUMNS)}"}
    sub = filter_seasons(stats(), season_from, season_to)
    if "gp" in sub.columns:
        sub = sub[sub.gp.fillna(0) >= min_games]
    if sub.empty:
        return {"error": "aucune donnée pour cette plage / ce filtre"}

    if aggregate:
        agg = (sub.groupby(["player_id", "player_name"], as_index=False)
                  .agg(**{stat: (stat, "mean"), "gp_total": ("gp", "sum"),
                          "seasons": ("season", "nunique")}))
        total = len(agg)
        agg = agg.sort_values(stat, ascending=False).head(limit)
        agg[stat] = agg[stat].round(2)
        return as_table(agg[["player_name", stat, "gp_total", "seasons"]],
                        total_players=total, players_returned=len(agg))

    sub2, n_kept, total = top_players_detailed(sub, stat, limit, ascending=False)
    cols = [c for c in ["player_name", "season", "team_abbreviation", stat, "gp"]
            if c in sub2.columns]
    return as_table(sub2[cols], total_players=total, players_returned=n_kept)


if __name__ == "__main__":
    mcp.run()
