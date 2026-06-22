"""Serveur MCP « nba-salaries » — salaires (source : dataset Kaggle nettoyé).

Expose les salaires joueur-saison de la base propre `data/clean/salaries_clean.parquet`
et la table de pont `data/clean/bridge.parquet` (lien player_id nba_api ↔ salaire).
Aucun appel réseau : tout est lu en local.

Lancement (test) :
    python mcp_servers/nba_salaries_server.py     # transport stdio

Outils exposés :
    get_player_salary(player, season=None)
    get_team_payroll(team, season)
    query_salaries(season, min_salary=0, top_n=20)
    list_seasons()
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
    """Pont player_id (nba_api) ↔ salaire, avec l'équipe issue des stats."""
    global _bridge
    if _bridge is None:
        if not BRIDGE_PATH.exists():
            raise FileNotFoundError(
                f"{BRIDGE_PATH} introuvable. Lance d'abord build_clean_datasets.py.")
        _bridge = pd.read_parquet(BRIDGE_PATH)
    return _bridge


def _start_year(season: str) -> int:
    """'2018-19' -> 2018."""
    return int(str(season).split("-")[0])


def as_table(df: pd.DataFrame) -> dict:
    """Format compact : {columns: [...], rows: [[...], ...]}.

    Bien plus court que des listes de dicts (les noms de colonnes ne sont écrits
    qu'une seule fois), donc plus facile à traiter pour le LLM.
    """
    cols = list(df.columns)
    rows = df.where(pd.notna(df), None).values.tolist()
    return {"columns": cols, "rows": rows}


def filter_seasons(df: pd.DataFrame,
                   season_from: str | None,
                   season_to: str | None) -> pd.DataFrame:
    """Restreint un DataFrame à une plage de saisons [season_from, season_to] incluses.

    Bornes optionnelles ; par défaut (None, None) = toute la plage disponible.
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

mcp = FastMCP("nba-salaries")


@mcp.tool()
def list_seasons() -> dict:
    """Liste les saisons couvertes par les salaires et les bornes de la plage totale.

    À appeler en premier pour connaître la période disponible (utile pour choisir
    `season_from` / `season_to`).
    """
    seasons = sorted(salaries().season.unique(), key=_start_year)
    return {
        "seasons": seasons,
        "season_from": seasons[0] if seasons else None,
        "season_to": seasons[-1] if seasons else None,
        "count": len(seasons),
    }


@mcp.tool()
def get_player_salary(player: str,
                      season_from: str | None = None,
                      season_to: str | None = None) -> dict:
    """Salaire d'un joueur sur une PLAGE de saisons.

    `player` = nom (ou fragment). `season_from` / `season_to` (ex. '2018-19')
    bornent la période, incluses ; par défaut = toute la plage disponible.
    Renvoie le salaire nominal et ajusté à l'inflation, une ligne par saison.
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
    """Masse salariale d'une équipe pour une saison (ex. team='LAL', season='2018-19').

    L'équipe provient de la table de pont (abréviation nba_api). Agrège les salaires
    des joueurs rattachés à cette équipe cette saison-là.
    """
    b = bridge()
    if "team_abbreviation" not in b.columns:
        return {"error": "team_abbreviation absent du pont ; régénère build_clean_datasets.py"}
    sub = b[(b.season == season) &
            (b.team_abbreviation.str.upper() == team.upper())]
    if sub.empty:
        return {"team": team, "season": season, "total_salary": 0, "players": 0}
    return {
        "team": team.upper(),
        "season": season,
        "total_salary": int(sub.salary.fillna(0).sum()),
        "players": int(sub.salary.notna().sum()),
    }


@mcp.tool()
def query_salaries(season_from: str | None = None,
                   season_to: str | None = None,
                   min_salary: int = 0,
                   aggregate: bool = False,
                   limit: int = 10) -> dict:
    """Salaires sur une PLAGE de saisons. Détaillé par défaut (1 ligne / joueur-saison).

    Trié par salaire décroissant : la 1re ligne est le mieux payé.

    - `season_from` / `season_to` (ex. '2018-19') bornent la période, incluses ;
      par défaut = toute la plage disponible.
    - `min_salary` filtre les salaires inférieurs au seuil.
    - `aggregate=False` (défaut) : renvoie chaque (joueur, saison) avec son salaire brut
      — idéal pour des visualisations (courbes, scatter).
    - `aggregate=True` : un résumé par joueur (salaire moyen + total + nb de saisons).
    - `limit` plafonne le nombre de lignes (défaut 10 ; augmente-le, p. ex. 500, pour
      préparer une visualisation).
    """
    sub = filter_seasons(salaries(), season_from, season_to)
    sub = sub[sub.salary.fillna(0) >= min_salary]
    if sub.empty:
        return {"error": "aucun salaire pour cette plage / ce filtre"}

    if aggregate:
        agg = (sub.groupby("name_norm", as_index=False)
                  .agg(player_name=("player_name", "first"),
                       salary_avg=("salary", "mean"),
                       salary_total=("salary", "sum"),
                       seasons=("season", "nunique")))
        agg = agg.sort_values("salary_avg", ascending=False).head(limit)
        agg["salary_avg"] = agg.salary_avg.round(0).astype("Int64")
        agg["salary_total"] = agg.salary_total.astype("Int64")
        return as_table(agg[["player_name", "salary_avg", "salary_total", "seasons"]])

    cols = [c for c in ["player_name", "season", "salary", "inflation_adj_salary"]
            if c in sub.columns]
    out = sub[cols].sort_values("salary", ascending=False).head(limit)
    return as_table(out)


@mcp.tool()
def get_value_ranking(metric: str = "pts",
                      season_from: str | None = None,
                      season_to: str | None = None,
                      min_games: int = 20,
                      order: str = "best",
                      aggregate: bool = False,
                      limit: int = 10) -> dict:
    """Rentabilité : performance rapportée au salaire, sur une PLAGE de saisons.

    Le résultat est TRIÉ : la 1re ligne est la plus rentable (order='best') ou la plus
    surpayée (order='worst'). Pour « le joueur le plus rentable », prends la 1re ligne.

    Croise stats et salaires (table de pont) et calcule, pour chaque ligne, le ratio
    `metric` par million de dollars : value = metric / (salary / 1_000_000).

    - `metric` ∈ {pts, reb, ast, stl, blk}.
    - `season_from` / `season_to` (ex. '2018-19') bornent la période, incluses ;
      par défaut = toute la plage disponible.
    - `min_games` écarte les saisons où le joueur a trop peu joué.
    - `order` = "best" (meilleurs rapports = bargains) ou "worst" (plus surpayés).
    - `aggregate=False` (défaut) : 1 ligne par (joueur, saison) avec metric, salary et
      value bruts — idéal pour visualiser (scatter salaire×perf, etc.).
    - `aggregate=True` : un résumé par joueur (moyennes sur la période).
    - `limit` plafonne le nombre de lignes (défaut 10, pour un classement court ;
      augmente-le, p. ex. 500, seulement pour préparer une visualisation).

    Outil à utiliser pour « le joueur le plus rentable », « les meilleurs contrats »,
    « les joueurs surpayés », et pour préparer les visualisations performance/salaire.
    """
    metric = metric.lower()
    allowed = {"pts", "reb", "ast", "stl", "blk"}
    if metric not in allowed:
        return {"error": f"metric inconnue '{metric}'. Choix : {', '.join(sorted(allowed))}"}

    b = bridge()
    needed = {metric, "gp", "salary", "player_name", "season"}
    if not needed.issubset(b.columns):
        return {"error": "colonnes manquantes dans bridge.parquet ; "
                         "régénère build_clean_datasets.py"}

    sub = filter_seasons(b, season_from, season_to).copy()
    sub = sub[(sub.salary.fillna(0) > 0) & (sub.gp.fillna(0) >= min_games)]
    if sub.empty:
        return {"error": f"aucune ligne avec salaire et gp >= {min_games} sur la période"}

    ascending = (order == "worst")

    if aggregate:
        agg = (sub.groupby(["player_id", "player_name"], as_index=False)
                  .agg(metric_avg=(metric, "mean"),
                       salary_avg=("salary", "mean"),
                       gp_total=("gp", "sum"),
                       seasons=("season", "nunique")))
        agg["value"] = (agg.metric_avg / (agg.salary_avg / 1_000_000)).round(3)
        agg["metric_avg"] = agg.metric_avg.round(1)
        agg["salary_avg"] = agg.salary_avg.round(0).astype("Int64")
        agg = agg.sort_values("value", ascending=ascending).head(limit)
        out = agg[["player_name", "metric_avg", "salary_avg", "value", "gp_total", "seasons"]]
        out = out.rename(columns={"metric_avg": f"{metric}_avg",
                                  "value": f"{metric}_per_million"})
        return as_table(out)

    # Détaillé : une ligne par (joueur, saison).
    sub["value"] = (sub[metric] / (sub.salary / 1_000_000)).round(3)
    sub = sub.sort_values("value", ascending=ascending).head(limit)
    out = sub[["player_name", "season", metric, "salary", "gp", "value"]]
    out = out.rename(columns={"value": f"{metric}_per_million"})
    return as_table(out)


if __name__ == "__main__":
    mcp.run()
