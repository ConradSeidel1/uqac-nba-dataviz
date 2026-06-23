# Assistant intelligent NBA — Performance vs. Contrats

Projet de visualisation de données (UQAC). L'application croise les **statistiques de jeu**
des joueurs NBA avec leurs **salaires** pour identifier les contrats les plus rentables et les
plus coûteux de la ligue. Elle combine un **tableau de bord interactif** et un **assistant
conversationnel** propulsé par un modèle de langage **exécuté en local**.

---

## 1. Aperçu

L'utilisateur cible est l'analyste *front office* d'une équipe NBA, qui doit construire le
meilleur effectif possible sous la contrainte du *salary cap*. L'application répond à ses
questions de deux façons complémentaires :

- **Un tableau de bord** (haut de la page) : quatre visualisations interactives qui répondent
  chacune à une question métier (rentabilité, budget vs résultats, ancienneté, valeur vs âge).
- **Un assistant conversationnel** (bas de la page) : on pose une question en langage naturel
  (« Quel est le joueur le plus rentable en 2021-22 ? ») et un agent interroge les données via
  des outils, puis répond avec des chiffres réels et sourcés — sans jamais les inventer.

Tout tourne **en local** : le modèle de langage via [Ollama](https://ollama.com), les données
en fichiers locaux. Aucune clé d'API, aucun service tiers, aucune donnée envoyée à l'extérieur.

---

## 2. Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │            Interface Streamlit (app.py)        │
                  │  ┌────────────────────┐   ┌─────────────────┐  │
                  │  │  Tableau de bord    │   │      Chat        │  │
                  │  │  (dashboard.py)     │   │  agent ReAct     │  │
                  │  │  4 graphes Plotly   │   │                  │  │
                  │  └─────────┬──────────┘   └────────┬─────────┘  │
                  └────────────┼───────────────────────┼───────────┘
                               │                        │
                   lecture directe            LLM local (Ollama) ─ LangGraph
                       Parquet                         │ outils via MCP
                               │              ┌─────────┴──────────┐
                               │              ▼                    ▼
                               │     ┌──────────────────┐ ┌──────────────────┐
                               │     │ Serveur MCP       │ │ Serveur MCP       │
                               │     │ « nba-stats »     │ │ « nba-salaries »  │
                               │     └────────┬─────────┘ └────────┬─────────┘
                               └──────────────┴────────────────────┘
                                              ▼
                                   data/clean/*.parquet
                              (bases propres + table de pont)
```

- **Agent ReAct** (LangChain / LangGraph) : raisonne en boucle *Thought → Action → Observation*.
  Il ne calcule jamais les chiffres lui-même ; il appelle des outils déterministes et commente
  leurs résultats. Cela limite fortement les hallucinations.
- **Deux serveurs MCP** ([Model Context Protocol](https://modelcontextprotocol.io)), un par
  source de données. L'application les lance automatiquement en sous-processus (transport
  stdio) et expose leurs outils à l'agent via `langchain-mcp-adapters`.
- **LLM local** via Ollama — aucune dépendance à un service en ligne ni à Claude Desktop.

---

## 3. Données

| Source | Contenu | Accès |
| --- | --- | --- |
| `nba_api` | Statistiques de jeu joueur-saison (points, rebonds, passes, %, bilan d'équipe…) | API officielle `stats.nba.com` |
| Kaggle | Salaires par joueur-saison (nominal + ajusté inflation) — dataset [*NBA Players and Team Data*](https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data), fichier `NBA Salaries (1990-2023)` | CSV local |

**Périmètre : saisons 1996-97 → 2021-22 (26 saisons)** — l'intersection réelle où les deux
sources fournissent des données (`nba_api` ne renvoie rien avant 1996-97 ; le dataset de
salaires s'arrête en 2021-22).

**Jointure.** Les deux sources n'ont pas d'identifiant commun : la jointure se fait par **nom
normalisé** (minuscules, accents et suffixes retirés) complétée d'une **table d'alias** manuelle
(`data/aliases.csv`) pour les cas particuliers. Les joueurs-saison sans salaire correspondant
sont **écartés** (pas de *fuzzy matching*, pour éviter les faux appariements). Résultat :
**11 659 lignes joueur-saison appariées** (2 322 joueurs), soit **94,7 %** du référentiel.

Le pipeline produit trois fichiers dans `data/clean/` :

- `stats_clean.parquet` — statistiques de jeu nettoyées ;
- `salaries_clean.parquet` — salaires nettoyés ;
- `bridge.parquet` — **table de pont** reliant chaque joueur-saison à son salaire et portant les
  colonnes dérivées (`team_tenure` = ancienneté dans l'équipe, bilan victoires, âge).

---

## 4. Installation

Prérequis : **Python 3.10+** et **[Ollama](https://ollama.com/download)** installé et lancé.

```bash
# 1. Dépendances Python
pip install -r requirements.txt

# 2. Modèle de langage local (au choix)
ollama pull qwen3:8b      # recommandé : bon appel d'outils, tient en ~5 Go de VRAM
# ollama pull gemma4:e2b  # alternative plus légère
```

### Reconstruire les données (optionnel)

Les bases propres sont déjà fournies dans `data/clean/`. Pour les régénérer depuis les sources :

```bash
python extract_nba_api.py        # extraction des stats (long : 26 saisons, rate limiting)
python build_clean_datasets.py   # nettoyage + jointure → data/clean/*.parquet
```

> Le dataset de salaires Kaggle (`data/NBA Salaries(1990-2023).csv`) doit être présent ;
> il est téléchargeable depuis le lien de la section Données.

---

## 5. Utilisation

```bash
streamlit run app.py                  # le modèle se décharge après 5 min d'inactivité
streamlit run app.py -- --keep-alive  # garde le modèle en mémoire (pratique en démo)
```

> Le `--` isolé est requis : Streamlit ne transmet au script que ce qui suit ce séparateur.
> En alternative au drapeau : `KEEP_ALIVE=1 streamlit run app.py`.

Au lancement, l'application **précharge le modèle** en mémoire (la barre latérale affiche
« Modèle prêt ✅ ») pour que la première question soit rapide.

### Le tableau de bord

Filtres communs dans la **barre latérale** : période (saison de début / de fin), métrique,
et sélection cherchable d'**équipes** et de **joueurs**. Quatre visualisations :

| Visualisation | Ce qu'elle montre | Question métier |
| --- | --- | --- |
| **Salaire × performance** (nuage de points) | 1 point par joueur (moyenne sur la période) + une droite de régression de référence : au-dessus = bonne affaire, en-dessous = surpayé | Joueurs rentables, *bargains* |
| **Masse salariale × victoires** (par équipe) | Le budget achète-t-il des victoires ? Ligne de tendance | Budget vs résultats |
| **Performance selon l'ancienneté** | Effet du nombre d'années consécutives dans la même équipe | Effet « durée de contrat » |
| **Rentabilité selon l'âge** | À quel âge un joueur offre le meilleur rapport performance/salaire | Valeur vs âge |

### L'assistant conversationnel

On pose une question en français ; l'agent choisit lui-même les outils, la métrique et la
période, puis répond avec des chiffres réels. Un panneau **« Étapes de raisonnement »** (replié
par défaut, déroulable) montre chaque appel d'outil et son résultat, sous forme de tableau.

Exemples de questions : *« Quel est le joueur le plus rentable ? »*, *« Les mieux payés en
2018-19 ? »*, *« Évolution du salaire de Stephen Curry »*, *« Quelles équipes ont la plus grosse
masse salariale en 2021-22 ? »*.

**Le chat pilote le tableau de bord.** Après chaque réponse, les filtres du dashboard se mettent
à jour automatiquement selon ce que l'agent a réellement consulté — déduit de ses **appels
d'outils** (source fiable), pas du texte. Seuls les filtres pertinents sont modifiés : la
métrique et la période depuis les arguments d'outil, le joueur ou l'équipe mis en avant depuis le
résultat. Demander « le joueur le plus rentable en 2021-22 » sélectionne ainsi automatiquement ce
joueur, la métrique et la saison dans les visualisations.

---

## 6. Structure du projet

```
.
├── app.py                      # application Streamlit (dashboard + chat + agent)
├── dashboard.py                # 4 visualisations Plotly + filtres
├── extract_nba_api.py          # extraction des stats depuis nba_api
├── build_clean_datasets.py     # nettoyage, jointure, table de pont
├── mcp_servers/
│   ├── nba_stats_server.py     # serveur MCP « nba-stats »
│   ├── nba_salaries_server.py  # serveur MCP « nba-salaries »
│   └── README.md               # détail des outils MCP
├── data/
│   ├── NBA Salaries(1990-2023).csv
│   ├── aliases.csv             # table d'alias pour la jointure par nom
│   └── clean/                  # bases propres générées (parquet)
└── requirements.txt
```

---

## 7. Choix techniques et limites

- **Le LLM n'invente jamais de chiffre.** Toute valeur provient d'un appel d'outil MCP ; les
  outils renvoient leurs résultats déjà triés, dans un format compact (colonnes + lignes) pour
  rester lisibles par un petit modèle local.
- **Jointure par nom.** En l'absence d'identifiant commun, c'est le risque principal ; il est
  maîtrisé par normalisation + table d'alias, et mesuré (94,7 % d'appariement, le reste écarté).
- **« Durée de contrat » approximée.** Faute de données contractuelles, elle est estimée par le
  nombre d'années consécutives d'un joueur dans la même équipe (ce qu'un spectateur appelle
  « depuis combien de temps il est dans l'équipe »).
- **Hors périmètre.** L'analyse par poste (meneur, ailier…) n'est pas couverte : le poste n'est
  pas fourni par les endpoints `nba_api` utilisés. La performance ne capture pas tout (défense,
  leadership, valeur marketing) : un contrat « surpayé » statistiquement peut être justifié
  autrement.

---

## 8. Pile technique

Python · Streamlit · Plotly · Pandas · LangChain / LangGraph · Model Context Protocol (FastMCP)
· Ollama · `nba_api`.
