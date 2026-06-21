"""
Extraction des données nba_api (référentiel du projet).

Récupère, pour chaque saison de 1990-91 à 2025-26 :
  1. les statistiques joueur-saison  (LeagueDashPlayerStats)
  2. les classements / bilans d'équipe (LeagueStandings)

et construit un référentiel maître des joueurs (player_id, nom, saison, équipe).

Caractéristiques :
  - extraction faite UNE SEULE FOIS en amont (cache local Parquet) ;
  - rate limiting + retry pour respecter stats.nba.com ;
  - reprise : une saison déjà extraite n'est pas re-téléchargée (sauf --force).

Usage :
    python extract_nba_api.py                  # toutes les saisons 1990-91 -> 2025-26
    python extract_nba_api.py --start 2018-19 --end 2020-21
    python extract_nba_api.py --force          # ré-extrait même si le cache existe
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueStandings
from requests.exceptions import ReadTimeout, ConnectionError as ReqConnectionError


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

FIRST_SEASON_START = 1990          # saison 1990-91 (couverture HoopsHype)
LAST_SEASON_START = 2025           # saison 2025-26 (saison courante)

LEAGUE_ID = "00"                   # 00 = NBA
SEASON_TYPE = "Regular Season"

# Politesse réseau : stats.nba.com applique un rate limiting agressif.
REQUEST_TIMEOUT = 60               # secondes par requête
DELAY_BETWEEN_REQUESTS = 1.5       # pause après chaque requête réussie
MAX_RETRIES = 4                    # tentatives par requête
RETRY_BACKOFF = 5                  # secondes, multiplié par le numéro de tentative

# Dossier de sortie : <racine_projet>/data/nba_api/
DATA_DIR = Path(__file__).resolve().parent / "data" / "nba_api"
STATS_DIR = DATA_DIR / "player_stats"
STANDINGS_DIR = DATA_DIR / "standings"
PLAYERS_REF_PATH = DATA_DIR / "players_ref.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_nba_api")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def season_label(start_year: int) -> str:
    """1990 -> '1990-91', 2025 -> '2025-26'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def all_seasons(start: int, end: int) -> list[str]:
    return [season_label(y) for y in range(start, end + 1)]


def _call_with_retry(endpoint_cls, label: str, **kwargs) -> pd.DataFrame:
    """Appelle un endpoint nba_api avec retry/backoff et renvoie le 1er DataFrame."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ep = endpoint_cls(timeout=REQUEST_TIMEOUT, **kwargs)
            df = ep.get_data_frames()[0]
            time.sleep(DELAY_BETWEEN_REQUESTS)
            return df
        except (ReadTimeout, ReqConnectionError, ValueError) as err:
            last_err = err
            wait = RETRY_BACKOFF * attempt
            log.warning(
                "%s : tentative %d/%d échouée (%s) — nouvelle tentative dans %ds",
                label, attempt, MAX_RETRIES, type(err).__name__, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Échec définitif sur {label}") from last_err


# --------------------------------------------------------------------------- #
# Extraction par saison
# --------------------------------------------------------------------------- #

def fetch_player_stats(season: str) -> pd.DataFrame:
    """Stats joueur-saison (totaux + de base) pour une saison donnée."""
    df = _call_with_retry(
        LeagueDashPlayerStats,
        label=f"player_stats {season}",
        season=season,
        season_type_all_star=SEASON_TYPE,
        league_id_nullable=LEAGUE_ID,
        per_mode_detailed="PerGame",
    )
    df.insert(0, "SEASON", season)
    return df


def fetch_standings(season: str) -> pd.DataFrame:
    """Classements / bilans victoires-défaites par équipe pour une saison."""
    df = _call_with_retry(
        LeagueStandings,
        label=f"standings {season}",
        season=season,
        season_type=SEASON_TYPE,
        league_id=LEAGUE_ID,
    )
    df.insert(0, "SEASON", season)
    return df


def build_players_ref_row(stats: pd.DataFrame, season: str) -> pd.DataFrame:
    """Extrait le sous-ensemble servant de référentiel maître des joueurs."""
    keep = [c for c in ("PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION")
            if c in stats.columns]
    ref = stats[keep].copy()
    ref.insert(0, "SEASON", season)
    return ref


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def extract_season(season: str, force: bool) -> pd.DataFrame:
    """Extrait (ou recharge depuis le cache) une saison ; renvoie le morceau de référentiel."""
    stats_path = STATS_DIR / f"player_stats_{season}.parquet"
    standings_path = STANDINGS_DIR / f"standings_{season}.parquet"

    if stats_path.exists() and standings_path.exists() and not force:
        log.info("%s : déjà en cache, ignorée (--force pour re-extraire)", season)
        stats = pd.read_parquet(stats_path)
        return build_players_ref_row(stats, season)

    log.info("%s : extraction en cours…", season)
    stats = fetch_player_stats(season)
    standings = fetch_standings(season)

    stats.to_parquet(stats_path, index=False)
    standings.to_parquet(standings_path, index=False)
    log.info("%s : %d joueurs, %d équipes enregistrés", season, len(stats), len(standings))

    return build_players_ref_row(stats, season)


def main() -> None:
    parser = argparse.ArgumentParser(description="Étape A — extraction nba_api")
    parser.add_argument("--start", default=season_label(FIRST_SEASON_START),
                        help="première saison, ex. 1990-91")
    parser.add_argument("--end", default=season_label(LAST_SEASON_START),
                        help="dernière saison, ex. 2025-26")
    parser.add_argument("--force", action="store_true",
                        help="ré-extrait même si la saison est déjà en cache")
    args = parser.parse_args()

    start_year = int(args.start.split("-")[0])
    end_year = int(args.end.split("-")[0])
    seasons = all_seasons(start_year, end_year)

    for d in (STATS_DIR, STANDINGS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("Périmètre : %s → %s (%d saisons)", seasons[0], seasons[-1], len(seasons))

    ref_parts: list[pd.DataFrame] = []
    failures: list[str] = []

    for season in seasons:
        try:
            ref_parts.append(extract_season(season, force=args.force))
        except Exception as err:  # une saison qui échoue ne bloque pas les autres
            log.error("%s : échec — %s", season, err)
            failures.append(season)

    if ref_parts:
        players_ref = pd.concat(ref_parts, ignore_index=True)
        players_ref.to_parquet(PLAYERS_REF_PATH, index=False)
        log.info("Référentiel maître : %d lignes joueur-saison → %s",
                 len(players_ref), PLAYERS_REF_PATH)

    if failures:
        log.warning("Saisons en échec (à relancer) : %s", ", ".join(failures))
    else:
        log.info("Extraction terminée sans échec.")


if __name__ == "__main__":
    main()
