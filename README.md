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
- **Salaires** — *scrapés depuis [HoopsHype](https://hoopshype.com/salaries/)*.
- **Périmètre** — saisons **1990-91 → 2025-26** (borné par la couverture HoopsHype), avec une
  jointure parfaite entre les deux sources sur le référentiel `nba_api`.

## Approche technique

Agent ReAct (LangChain) connecté à deux serveurs MCP — un par source de données — avec un LLM
local, une interface Streamlit (chat + tableau de bord) et des visualisations Plotly.
