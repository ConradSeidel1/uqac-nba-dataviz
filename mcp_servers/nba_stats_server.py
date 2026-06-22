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


def _start_year(season: str) -> int:
    """'2018-19' -> 2018."""
    return int(str(season).split("-")[0])


def as_table(df: pd.DataFrame) -> dict:
    """Format compact : {columns: [...], rows: [[...], ...]}.

    Bien plus court que des listes de dicts (les noms de colonnes ne sont écrits
    qu'une seule fois), donc plus facile à traiter pour le LLM. Les valeurs sont
    converties en types JSON simples.
    """
    cols = list(df.columns)
    rows = df.where(pd.notna(df), None).values.tolist()
    return {"columns": cols, "rows": rows}


def filter_seasons(df: pd.DataFrame,
                   season_from: str | None,
                   season_to: str | None) -> pd.DataFrame:
    """Restreint un DataFrame à une plage de saisons [season_from, season_to].

    Les deux bornes sont incluses et optionnelles. Par défaut (None, None) =
    toute la plage disponible. Les bornes acceptent le format '2018-19'.
    """
    out = df
    if season_from:
        out = out[out["season"].map(_start_year) >= _start_year(season_from)]
    if season_to:
        out = out[out["season"].map(_start_year) <= _start_year(season_to)]
    return out


# --------------------------------------------------------------------------- #
# Serveur
# --------------------------------------------------------------------------- #

mcp = FastMCP("nba-stats")


@mcp.tool()
def list_seasons() -> dict:
    """Liste les saisons disponibles et les bornes de la plage totale.

    À appeler en premier pour connaître la période couverte (utile pour choisir
    `season_from` / `season_to` dans les autres outils).
    """
    seasons = sorted(stats().season.unique(), key=_start_year)
    return {
        "seasons": seasons,
        "season_from": seasons[0] if seasons else None,
        "season_to": seasons[-1] if seasons else None,
        "count": len(seasons),
    }


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
def get_player_season_stats(player: str,
                            season_from: str | None = None,
                            season_to: str | None = None) -> dict:
    """Stats d'un joueur sur une PLAGE de saisons.

    `player` = nom (ou fragment). `season_from` / `season_to` (ex. '2018-19')
    bornent la période, incluses ; par défaut = toute la plage disponible.
    Renvoie un tableau compact {columns, rows} : une ligne par saison du joueur.
    """
    sub = stats()
    sub = sub[sub.name_norm.str.contains(_norm(player), regex=False)]
    sub = filter_seasons(sub, season_from, season_to)
    cols = ["season", "player_id", "player_name", "team_abbreviation",
            "age", "gp", "min", "pts", "reb", "ast", "stl", "blk", "tov",
            "fg_pct", "fg3_pct", "ft_pct", "plus_minus"]
    cols = [c for c in cols if c in sub.columns]
    out = sub[cols].sort_values("season", key=lambda s: s.map(_start_year))
    return as_table(out)


@mcp.tool()
def get_league_leaders(stat: str,
                       season_from: str | None = None,
                       season_to: str | None = None,
                       min_games: int = 0,
                       aggregate: bool = False,
                       limit: int = 10) -> dict:
    """Meilleurs joueurs pour une statistique sur une PLAGE de saisons.

    Renvoie un tableau compact {columns, rows}, trié par stat décroissante :
    la 1re ligne de `rows` est le meilleur.

    `stat` ∈ {pts, reb, ast, stl, blk, tov, min, gp, fg_pct, fg3_pct, ft_pct,
    plus_minus, age}.
    - `season_from` / `season_to` (ex. '2018-19') bornent la période, incluses ;
      par défaut = toute la plage disponible.
    - `min_games` écarte les saisons où le joueur a trop peu joué.
    - `aggregate=False` (défaut) : 1 ligne par (joueur, saison) avec la stat brute —
      idéal pour visualiser (courbes, scatter).
    - `aggregate=True` : un résumé par joueur (moyenne de la stat sur la période).
    - `limit` plafonne le nombre de lignes (défaut 10 ; augmente-le, p. ex. 500, pour
      préparer une visualisation).
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
                  .agg(**{stat: (stat, "mean"),
                          "gp_total": ("gp", "sum"),
                          "seasons": ("season", "nunique")}))
        agg = agg.sort_values(stat, ascending=False).head(limit)
        agg[stat] = agg[stat].round(2)
        return as_table(agg[["player_name", stat, "gp_total", "seasons"]])

    cols = [c for c in ["player_name", "season", "team_abbreviation", stat, "gp"]
            if c in sub.columns]
    out = sub[cols].sort_values(stat, ascending=False).head(limit)
    return as_table(out)


if __name__ == "__main__":
    mcp.run()
