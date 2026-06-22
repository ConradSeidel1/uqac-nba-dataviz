"""Serveur MCP « nba-stats » — statistiques de jeu (source : nba_api nettoyée).

Expose les statistiques joueur-saison de la base propre `data/clean/stats_clean.parquet`
(produite par build_clean_datasets.py). Aucun appel réseau : tout est lu en local.

Lancement (test) :
    python mcp_servers/nba_stats_server.py        # transport stdio

Outils exposés :
    get_player_season_stats(player, season=None)
    get_league_leaders(stat, season, top_n=10)
    list_seasons()
    search_players(query, limit=20)
"""

from __future__ import annotations

import unicodedata
import re
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #

DATA = Path(__file__).resolve().parent.parent / "data" / "clean"
STATS_PATH = DATA / "stats_clean.parquet"

# Statistiques exposées et leur libellé.
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


# --------------------------------------------------------------------------- #
# Serveur
# --------------------------------------------------------------------------- #

mcp = FastMCP("nba-stats")


@mcp.tool()
def list_seasons() -> list[str]:
    """Liste les saisons disponibles (ex. '2018-19'), triées."""
    return sorted(stats().season.unique(), key=lambda s: int(s.split("-")[0]))


@mcp.tool()
def search_players(query: str, limit: int = 20) -> list[dict]:
    """Recherche des joueurs par nom (insensible aux accents/casse).

    Renvoie une liste de {player_id, player_name, seasons} pour les correspondances.
    """
    df = stats()
    q = _norm(query)
    hit = df[df.name_norm.str.contains(q, regex=False)]
    out = []
    for pid, g in hit.groupby("player_id"):
        out.append({
            "player_id": int(pid),
            "player_name": g.player_name.iloc[0],
            "seasons": sorted(g.season.unique(), key=lambda s: int(s.split("-")[0])),
        })
        if len(out) >= limit:
            break
    return out


@mcp.tool()
def get_player_season_stats(player: str, season: str | None = None) -> list[dict]:
    """Stats d'un joueur. `player` = nom (ou fragment). `season` ex. '2018-19' (optionnel).

    Si la saison est omise, renvoie toutes les saisons du joueur.
    """
    df = stats()
    sub = df[df.name_norm.str.contains(_norm(player), regex=False)]
    if season:
        sub = sub[sub.season == season]
    cols = ["season", "player_id", "player_name", "team_abbreviation",
            "age", "gp", "min", "pts", "reb", "ast", "stl", "blk", "tov",
            "fg_pct", "fg3_pct", "ft_pct", "plus_minus"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].to_dict("records")


@mcp.tool()
def get_league_leaders(stat: str, season: str, top_n: int = 10) -> list[dict]:
    """Meilleurs joueurs pour une statistique sur une saison donnée.

    `stat` ∈ {pts, reb, ast, stl, blk, tov, min, gp, fg_pct, fg3_pct, ft_pct,
    plus_minus, age}. `season` ex. '2018-19'.
    """
    stat = stat.lower()
    if stat not in STAT_COLUMNS:
        return [{"error": f"stat inconnue '{stat}'. Choix : {', '.join(STAT_COLUMNS)}"}]
    df = stats()
    sub = df[df.season == season].copy()
    if sub.empty:
        return [{"error": f"aucune donnée pour la saison {season}"}]
    sub = sub.sort_values(stat, ascending=False).head(top_n)
    return sub[["player_name", "team_abbreviation", stat, "gp", "min"]].to_dict("records")


if __name__ == "__main__":
    mcp.run()
