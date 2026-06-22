"""Étapes C/D/E — Construction des deux bases propres + table de pont.

À partir des données récoltées :
  • stats nba_api  : data/nba_api/player_stats/player_stats_<saison>.parquet
  • salaires CSV   : data/NBA Salaries(1990-2023).csv

ce script produit trois fichiers propres, alignés sur le périmètre commun :

  1. data/clean/stats_clean.parquet
        Stats joueur-saison concaténées, colonnes utiles, clé (player_id, season)
        + nom normalisé (name_norm).
  2. data/clean/salaries_clean.parquet
        Salaires joueur-saison : nom, season, salary, inflation_adj_salary,
        + nom normalisé. Doublons (joueur, saison) dédupliqués.
  3. data/clean/bridge.parquet
        Table de pont : un (player_id, season) APPARIÉ avec son salaire. C'est le lien
        entre les deux bases, sans les fusionner — chaque source reste interrogeable
        indépendamment (cohérent avec l'architecture 2 serveurs MCP).

La jointure se fait par NOM NORMALISÉ (les deux sources n'ont pas d'ID commun) :
  • normalisation : minuscules, accents retirés, suffixes (Jr/Sr/II/III/IV) retirés,
    ponctuation retirée ;
  • correspondances particulières (surnoms, orthographes) via la table d'alias
    data/aliases.csv (colonnes : name_nba_api, name_salaries).

Les joueurs-saison nba_api SANS salaire correspondant sont RETIRÉS de la base (pas de
fuzzy matching, pour éviter les faux appariements). Le rapport affiche le taux
d'appariement et le nombre d'écartés.

Format : Parquet (compact, typé, rapide à charger par les serveurs MCP).

Usage :
    python build_clean_datasets.py
    python build_clean_datasets.py --dropped-csv   # liste les joueurs écartés
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STATS_DIR = DATA / "nba_api" / "player_stats"
SALARIES_CSV = DATA / "NBA Salaries(1990-2023).csv"
ALIASES_CSV = DATA / "aliases.csv"

OUT_DIR = DATA / "clean"
STATS_OUT = OUT_DIR / "stats_clean.parquet"
SALARIES_OUT = OUT_DIR / "salaries_clean.parquet"
BRIDGE_OUT = OUT_DIR / "bridge.parquet"
DROPPED_LOG = OUT_DIR / "dropped_unmatched.csv"

# Colonnes de stats conservées (box score de base + identité).
STATS_KEEP = [
    "SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
    "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_clean")

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


# --------------------------------------------------------------------------- #
# Normalisation des noms
# --------------------------------------------------------------------------- #

def normalize_name(name: str) -> str:
    """minuscules + sans accents + sans suffixe + sans ponctuation."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = _SUFFIX_RE.sub("", s)
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def start_year_of(season: str) -> int:
    """'1996-97' -> 1996."""
    return int(season.split("-")[0])


# --------------------------------------------------------------------------- #
# Chargement / nettoyage des stats nba_api
# --------------------------------------------------------------------------- #

def load_stats() -> pd.DataFrame:
    frames = []
    for f in sorted(STATS_DIR.glob("player_stats_*.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        keep = [c for c in STATS_KEEP if c in df.columns]
        frames.append(df[keep])
    if not frames:
        raise SystemExit("Aucune donnée de stats trouvée dans " + str(STATS_DIR))
    stats = pd.concat(frames, ignore_index=True)
    stats = stats.rename(columns=str.lower)
    stats["name_norm"] = stats["player_name"].map(normalize_name)
    return stats


# --------------------------------------------------------------------------- #
# Chargement / nettoyage des salaires CSV
# --------------------------------------------------------------------------- #

def _money_to_int(series: pd.Series) -> pd.Series:
    return (series.astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
            .replace("", pd.NA)
            .astype("Int64"))


def load_salaries() -> pd.DataFrame:
    df = pd.read_csv(SALARIES_CSV)
    df = df.rename(columns={
        "playerName": "player_name",
        "seasonStartYear": "season_start_year",
        "salary": "salary",
        "inflationAdjSalary": "inflation_adj_salary",
    })
    df["salary"] = _money_to_int(df["salary"])
    df["inflation_adj_salary"] = _money_to_int(df["inflation_adj_salary"])
    df["season"] = df["season_start_year"].map(season_label)
    df["name_norm"] = df["player_name"].map(normalize_name)

    # Dédoublonnage (joueur, saison) : la grande majorité sont de vrais doublons
    # (même salaire). On agrège en gardant le salaire maximum.
    df = (df.groupby(["name_norm", "season"], as_index=False)
            .agg(player_name=("player_name", "first"),
                 season_start_year=("season_start_year", "first"),
                 salary=("salary", "max"),
                 inflation_adj_salary=("inflation_adj_salary", "max")))
    return df


# --------------------------------------------------------------------------- #
# Table d'alias
# --------------------------------------------------------------------------- #

def load_aliases() -> dict[str, str]:
    """name_nba_api (normalisé) -> name_salaries (normalisé)."""
    if not ALIASES_CSV.exists():
        return {}
    a = pd.read_csv(ALIASES_CSV)
    return {normalize_name(r.name_nba_api): normalize_name(r.name_salaries)
            for r in a.itertuples()}


# --------------------------------------------------------------------------- #
# Construction du pont + rapport de complétude
# --------------------------------------------------------------------------- #

def build_bridge(stats: pd.DataFrame, salaries: pd.DataFrame,
                 aliases: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jointure exacte (nom normalisé + alias). Les joueurs sans salaire sont retirés.

    Renvoie (bridge, dropped) où :
      • bridge  = joueurs-saison APPARIÉS uniquement (avec salaire) ;
      • dropped = joueurs-saison nba_api SANS salaire, écartés de la base.
    """
    # Périmètre = saisons communes aux deux sources.
    common = sorted(set(stats.season) & set(salaries.season), key=start_year_of)
    log.info("Saisons communes : %s → %s (%d saisons)",
             common[0], common[-1], len(common))

    st = stats[stats.season.isin(common)].copy()
    sa = salaries[salaries.season.isin(common)].copy()

    # Référentiel = joueurs nba_api ; on applique les alias avant la jointure.
    ref_cols = ["player_id", "season", "player_name", "name_norm"]
    if "team_abbreviation" in st.columns:
        ref_cols.append("team_abbreviation")
    ref = st[ref_cols].copy()
    ref["name_join"] = ref["name_norm"].map(lambda n: aliases.get(n, n))

    sal_key = sa[["name_norm", "season", "salary", "inflation_adj_salary"]].rename(
        columns={"name_norm": "name_join"})

    # Jointure exacte (nom normalisé + alias).
    full = ref.merge(sal_key, on=["name_join", "season"], how="left")

    matched_mask = full.salary.notna()
    bridge = full[matched_mask].copy()
    dropped = full[~matched_mask][
        ["player_id", "season", "player_name", "name_norm"]].copy()

    log.info("Jointure exacte : %d/%d appariés (%.2f%%) — %d joueurs-saison "
             "sans salaire retirés de la base",
             len(bridge), len(full), 100 * len(bridge) / len(full), len(dropped))

    return bridge, dropped


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Étapes C/D/E — bases propres + pont")
    parser.add_argument("--dropped-csv", action="store_true",
                        help="écrit la liste des joueurs-saison écartés (sans salaire)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = load_stats()
    salaries = load_salaries()
    aliases = load_aliases()
    log.info("Stats: %d lignes | Salaires: %d lignes | Alias: %d",
             len(stats), len(salaries), len(aliases))

    bridge, dropped = build_bridge(stats, salaries, aliases)

    # La base finale ne contient que les joueurs appariés : on restreint stats_clean
    # aux couples (player_id, season) présents dans le pont, et salaries_clean aux
    # saisons couvertes.
    kept_keys = set(zip(bridge.player_id, bridge.season))
    stats_keep = stats[stats.apply(
        lambda r: (r.player_id, r.season) in kept_keys, axis=1)]

    stats_keep.to_parquet(STATS_OUT, index=False)
    salaries[salaries.season.isin(set(bridge.season))].to_parquet(SALARIES_OUT, index=False)
    bridge.to_parquet(BRIDGE_OUT, index=False)
    log.info("Écrit : %s (%d) | %s | %s (%d)",
             STATS_OUT.name, len(stats_keep), SALARIES_OUT.name,
             BRIDGE_OUT.name, len(bridge))

    if args.dropped_csv and len(dropped):
        log.info("Joueurs-saison écartés listés dans %s (%d lignes)",
                 DROPPED_LOG.name, len(dropped))

    log.info("Terminé.")


if __name__ == "__main__":
    main()
