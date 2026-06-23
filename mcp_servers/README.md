# Serveurs MCP — nba-stats & nba-salaries

Deux serveurs [FastMCP](https://github.com/modelcontextprotocol/python-sdk) (transport stdio),
un par source de données, lisant les bases propres de `data/clean/`.

| Serveur | Fichier | Lit | Outils |
| --- | --- | --- | --- |
| `nba-stats` | `nba_stats_server.py` | `stats_clean.parquet` | `get_player_season_stats`, `get_league_leaders`, `search_players`, `list_seasons` |
| `nba-salaries` | `nba_salaries_server.py` | `salaries_clean.parquet`, `bridge.parquet` | `get_player_salary`, `get_team_payroll`, `query_salaries`, `get_value_ranking`, `list_seasons` |

## Prérequis

```bash
pip install -r ../requirements.txt   # inclut mcp[cli], pandas, pyarrow
python ../build_clean_datasets.py     # génère data/clean/*.parquet
```

## Tester un serveur isolément

```bash
# Inspecteur interactif (visualise les outils et leurs réponses)
mcp dev nba_stats_server.py

# Ou lancement direct (stdio, attend un client)
python nba_stats_server.py
```

## Utilisation dans le projet

Les serveurs ne se branchent **pas** sur Claude Desktop. Ils sont consommés par l'agent
ReAct du projet (LLM **local** via Ollama) à travers `langchain-mcp-adapters` : l'app
`../app.py` les lance automatiquement en sous-processus stdio et expose leurs outils à
l'agent. Voir la racine du dépôt pour lancer l'application (`streamlit run app.py`).
