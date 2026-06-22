# uqac-nba-dataviz

**Assistant intelligent NBA : Performance vs. Contrats**

Projet de visualisation de données (UQAC) qui croise les **statistiques de jeu** des joueurs NBA
avec leurs **salaires** pour identifier les contrats les plus rentables et les plus coûteux de la ligue.

## Objectif

Construire un assistant en langage naturel destiné à un analyste *front office* d'équipe NBA :
poser une question (« qui sont les joueurs les plus rentables ? ») et obtenir une réponse
chiffrée, sourcée et accompagnée d'une visualisation — sans écrire de code.

## Données

- **Statistiques de jeu** — extraites de l'API officielle via `nba_api`.
- **Salaires** — dataset Kaggle [*NBA Players and Team Data*](https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data)
  (fichier `NBA Salaries (1990-2023)`).
- **Périmètre** — saisons **1996-97 → 2021-22** (intersection réelle où les deux sources
  ont des données), avec une jointure entre les deux sur le référentiel `nba_api`.

## Approche technique

Agent ReAct (LangChain / LangGraph) connecté à deux serveurs MCP — un par source de données —
piloté par un LLM **local** (Ollama), avec une interface Streamlit qui combine un **tableau de
bord de visualisations** (Plotly) et un **chat**. Aucune dépendance à Claude Desktop : les
serveurs MCP sont lancés en sous-processus stdio par l'application.

## Avancement

- **Extraction `nba_api`** (`extract_nba_api.py`) → `data/nba_api/`. Les saisons antérieures à
  1996-97 ne sont pas renvoyées par l'API ; le périmètre réel démarre donc en 1996-97.
- **Salaires** : dataset Kaggle `NBA Salaries (1990-2023)` (`data/NBA Salaries(1990-2023).csv`).
- **Nettoyage + jointure** (`build_clean_datasets.py`) : deux bases propres et une table de pont
  dans `data/clean/` — `stats_clean.parquet`, `salaries_clean.parquet`, `bridge.parquet`.
  Jointure par nom normalisé + table d'alias (`data/aliases.csv`) ; les joueurs-saison sans
  salaire correspondant sont écartés.
- **Serveurs MCP** `nba-stats` et `nba-salaries` (`mcp_servers/`) lisant `data/clean/`.
- **Agent ReAct + chat Streamlit** (`app.py`) : LLM local (Ollama) consommant les deux serveurs
  MCP via `langchain-mcp-adapters`.
- **Tableau de bord** (`dashboard.py`) : 4 visualisations Plotly affichées au-dessus du chat,
  chacune répondant à une question métier (voir ci-dessous).

## Reproduire le pipeline

```bash
pip install -r requirements.txt
ollama pull qwen3:8b             # LLM local conseillé (tient en ~5 Go, bon tool-calling)
# alternative testée : ollama pull gemma4:e2b

python extract_nba_api.py        # extraction stats (long : ~30 saisons, rate limiting)
python build_clean_datasets.py   # bases propres + pont

streamlit run app.py             # chat : agent ReAct + Ollama + serveurs MCP
```

## Lancer l'assistant (`app.py`)

```bash
streamlit run app.py                  # modèle déchargé après 5 min d'inactivité
streamlit run app.py -- --keep-alive  # modèle gardé en mémoire (pratique en démo)
```

Le `--` isolé est requis : Streamlit ne transmet au script que ce qui suit ce séparateur.

L'application :

- **Précharge le modèle** (warmup) au démarrage : on paie le chargement Ollama tout de
  suite plutôt qu'à la première question. La barre latérale affiche « Modèle prêt ✅ ».
- **`--keep-alive`** demande à Ollama de garder le modèle en VRAM indéfiniment. Le modèle
  vit dans le service Ollama (pas dans le process Streamlit) : il **survit à un `Ctrl+C`**
  de l'app. Pour libérer la VRAM ensuite : `ollama stop qwen3:8b`. Sans ce drapeau, Ollama
  décharge le modèle après 5 min d'inactivité (défaut).
- **Étapes de raisonnement** : une case dans la barre latérale affiche en direct, sous
  chaque réponse, les appels d'outils MCP de l'agent (Thought → Action → Observation),
  rendus sous forme de tableaux lisibles.
- Le **modèle est configurable** dans la barre latérale (champ « Modèle Ollama »), pour
  comparer p. ex. `qwen3:8b` et `gemma4:e2b` sans relancer.

## Tableau de bord & questions métier (`dashboard.py`)

Quatre visualisations s'affichent **au-dessus du chat**, chacune avec ses propres filtres
(saison et/ou métrique) et reliée à une question métier :

| Visualisation | Filtres | Question métier |
| --- | --- | --- |
| Scatter **salaire × performance** | saison, métrique | Q1 / Q5 — joueurs les plus rentables, détection des *bargains* |
| **Masse salariale × victoires** par équipe (avec tendance) | saison | Q2 — le budget achète-t-il des victoires ? |
| Performance selon l'**ancienneté dans l'équipe** | saison, métrique | Q4 — effet « durée de contrat » |
| **Rentabilité (perf/M$) selon l'âge** | métrique | Q6 — à quel âge un joueur offre le meilleur rapport |

Notes :

- La « durée de contrat » (Q4) est approximée par le nombre d'**années consécutives** d'un
  joueur dans la même équipe (colonne `team_tenure`, calculée dans `build_clean_datasets.py`).
- Le **bilan victoires** (Q2) et l'**âge** (Q6) sont embarqués dans `bridge.parquet`.
- Q3 (postes surpayés/sous-payés) est **hors périmètre** : le poste n'est pas fourni par les
  endpoints `nba_api` utilisés.
