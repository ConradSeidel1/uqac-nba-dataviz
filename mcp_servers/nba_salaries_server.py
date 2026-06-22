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


# --------------------------------------------------------------------------- #
# Serveur
# --------------------------------------------------------------------------- #

mcp = FastMCP("nba-salaries")


@mcp.tool()
def list_seasons() -> list[str]:
    """Liste les saisons couvertes par les salaires (ex. '2018-19'), triées."""
    return sorted(salaries().season.unique(), key=lambda s: int(s.split("-")[0]))


@mcp.tool()
def get_player_salary(player: str, season: str | None = None) -> list[dict]:
    """Salaire d'un joueur. `player` = nom (ou fragment), `season` ex. '2018-19' (optionnel).

    Renvoie le salaire nominal et le salaire ajusté à l'inflation par saison.
    """
    df = salaries()
    sub = df[df.name_norm.str.contains(_norm(player), regex=False)]
    if season:
        sub = sub[sub.season == season]
    cols = [c for c in ["player_name", "season", "salary", "inflation_adj_salary"]
            if c in sub.columns]
    return (sub[cols]
            .sort_values("season", key=lambda s: s.map(lambda x: int(x.split("-")[0])))
            .to_dict("records"))


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
def query_salaries(season: str, min_salary: int = 0, top_n: int = 20) -> list[dict]:
    """Salaires d'une saison, filtrés (min_salary) et triés décroissant (top_n).

    Utile pour « les mieux payés de 2018-19 », « les contrats > 30 M$ », etc.
    """
    df = salaries()
    sub = df[(df.season == season) & (df.salary.fillna(0) >= min_salary)]
    sub = sub.sort_values("salary", ascending=False).head(top_n)
    cols = [c for c in ["player_name", "season", "salary", "inflation_adj_salary"]
            if c in sub.columns]
    return sub[cols].to_dict("records")


if __name__ == "__main__":
    mcp.run()
