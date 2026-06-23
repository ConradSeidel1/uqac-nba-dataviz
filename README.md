# Assistant intelligent NBA — Performance vs. Contrats

> Projet de visualisation de données — UQAC.
> Une application qui croise les **statistiques de jeu** des joueurs NBA avec leurs
> **salaires** pour évaluer la rentabilité des contrats, à travers un **tableau de bord
> interactif** et un **assistant conversationnel** propulsé par un modèle de langage
> **exécuté entièrement en local**.

---

## Table des matières

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Questions métier](#2-questions-métier)
3. [Architecture générale](#3-architecture-générale)
4. [Les données et leur préparation](#4-les-données-et-leur-préparation)
5. [Les serveurs MCP et leurs outils](#5-les-serveurs-mcp-et-leurs-outils)
6. [L'agent conversationnel](#6-lagent-conversationnel)
7. [Le tableau de bord](#7-le-tableau-de-bord)
8. [Installation](#8-installation)
9. [Utilisation](#9-utilisation)
10. [Structure du projet](#10-structure-du-projet)
11. [Difficultés rencontrées et limites](#11-difficultés-rencontrées-et-limites)
12. [Pile technique](#12-pile-technique)

---

## 1. Contexte et problématique

En NBA, chaque équipe construit son effectif sous la contrainte d'un **plafond salarial**
(*salary cap*, ~136 M$ en 2022-23). Tout dollar mal investi dans un contrat est un dollar
indisponible pour renforcer l'équipe ailleurs. L'enjeu central d'un *front office* est donc
d'identifier les contrats qui **valent leur prix** — et ceux qui ne les valent pas.

Le problème : les écarts de salaire sont énormes (le contrat le plus cher de notre période
atteint 48 M$, contre un salaire médian autour de 3,7 M$, soit un facteur 13), et la
performance sportive ne se lit pas directement sur la fiche de paie. Évaluer la rentabilité
suppose de **croiser deux mondes** habituellement séparés : les statistiques de jeu (la valeur
sportive) et les salaires (la valeur contractuelle).

**L'utilisateur cible** est l'analyste *front office*, bras droit du General Manager. Il a
besoin de réponses **rapides, chiffrées et sourcées** pour préparer drafts, échanges et
re-signatures — sans écrire lui-même de code ni de requêtes. L'application répond à ce besoin
de deux manières complémentaires :

- un **tableau de bord** pour explorer visuellement les tendances ;
- un **assistant** auquel on pose des questions en langage naturel, qui interroge les données
  et répond avec des chiffres réels.

L'ensemble s'exécute **en local** : modèle de langage, données, calculs. Aucune clé d'API,
aucun service tiers, aucune donnée envoyée à l'extérieur.

---

## 2. Questions métier

Le projet est structuré autour de six questions métier. Le tableau ci-dessous indique, pour
chacune, comment elle est traitée — et assume honnêtement celles qui sortent du périmètre des
données disponibles.

| # | Question | Traitement |
| --- | --- | --- |
| **Q1** | Quels joueurs offrent le meilleur ratio performance / salaire ? | Outil `get_value_ranking` + visualisation salaire × performance |
| **Q2** | La masse salariale d'une équipe achète-t-elle des victoires ? | Visualisation masse salariale × victoires (avec tendance) |
| **Q3** | Quels postes sont structurellement surpayés ? | **Hors périmètre** : le poste n'est pas fourni par l'API utilisée |
| **Q4** | L'ancienneté dans l'équipe change-t-elle le rendement ? | Visualisation performance × ancienneté (« durée de contrat ») |
| **Q5** | Quels joueurs performants sont relativement sous-payés (*bargains*) ? | `get_value_ranking` + lecture des quadrants du nuage de points |
| **Q6** | À quel âge un joueur offre-t-il le meilleur rapport qualité/prix ? | Visualisation rentabilité × âge |

Les six questions couvrent les types d'analyse attendus : statistiques descriptives (Q1, Q3),
corrélations et comparaisons (Q2, Q4, Q6), et détection de phénomènes particuliers (Q5).

---

## 3. Architecture générale

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

L'application repose sur trois briques :

- **L'interface Streamlit** (`app.py`) assemble le tableau de bord et le chat sur une seule
  page. Le tableau de bord lit directement les fichiers Parquet ; le chat passe par l'agent.
- **L'agent ReAct** (LangGraph) raisonne en boucle *Thought → Action → Observation* sur un LLM
  local. Il ne calcule jamais les chiffres lui-même : il choisit et appelle des **outils
  déterministes**, observe leurs résultats, puis formule sa réponse. Ce découplage est la
  principale défense contre les hallucinations.
- **Deux serveurs MCP** ([Model Context Protocol](https://modelcontextprotocol.io)), un par
  source de données, exposent ces outils. L'application les lance automatiquement en
  sous-processus (transport stdio) et les connecte à l'agent via `langchain-mcp-adapters`.

Le choix du **MCP** est volontairement pédagogique : il fournit une interface standardisée
entre l'agent et les données, où chaque serveur est isolé, réutilisable (testable seul dans un
inspecteur MCP) et facilement remplaçable.

---

## 4. Les données et leur préparation

### 4.1 Les deux sources

| Source | Contenu | Accès |
| --- | --- | --- |
| `nba_api` | Statistiques de jeu joueur-saison : points, rebonds, passes, interceptions, contres, pourcentages de tir, bilan victoires/défaites de l'équipe… | API officielle `stats.nba.com` |
| Kaggle | Salaires par joueur-saison, nominal **et ajusté à l'inflation** — dataset [*NBA Players and Team Data*](https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data), fichier `NBA Salaries (1990-2023)` | CSV local |

La complémentarité est réelle : **aucune des deux sources ne suffit seule**. Les stats donnent
la valeur sportive, les salaires la valeur contractuelle ; seule leur jointure permet de
calculer un ratio « valeur pour l'argent ».

### 4.2 Le périmètre

**Saisons 1996-97 → 2021-22, soit 26 saisons.** Cette borne n'est pas arbitraire : c'est
l'**intersection réelle** où les deux sources fournissent des données. L'endpoint `nba_api`
utilisé ne renvoie rien avant 1996-97, et le dataset de salaires s'arrête à la saison 2021-22.
On compte environ **432 joueurs par saison**.

### 4.3 Le pipeline de préparation

Le pipeline se déroule en trois temps, exécutés une seule fois en amont :

1. **Extraction** (`extract_nba_api.py`) — interroge `nba_api` saison par saison, avec gestion
   du *rate limiting* (l'API limite les requêtes rapprochées), et stocke les statistiques
   brutes en Parquet local. Cela évite tout appel réseau pendant l'utilisation de l'application.

2. **Nettoyage et jointure** (`build_clean_datasets.py`) — c'est le cœur technique. Les deux
   sources **n'ont aucun identifiant commun** : la jointure ne peut se faire que par le nom du
   joueur, ce qui est le risque principal du projet. La stratégie retenue :
   - **normalisation des noms** des deux côtés (minuscules, accents et suffixes Jr./III/…
     retirés, ponctuation supprimée) ;
   - une **table d'alias manuelle** (`data/aliases.csv`) pour les cas particuliers (surnoms,
     orthographes divergentes) ;
   - les joueurs-saison **sans salaire correspondant sont écartés** — on a délibérément renoncé
     au *fuzzy matching* automatique, qui produisait de faux appariements (ex. « Mo Williams »
     confondu avec « Monty Williams »).

   Résultat mesuré : **11 659 lignes joueur-saison appariées** sur 12 307, soit **94,7 %** du
   référentiel, sans aucun faux appariement.

3. **Sorties** — trois fichiers Parquet dans `data/clean/` :
   - `stats_clean.parquet` : statistiques de jeu nettoyées ;
   - `salaries_clean.parquet` : salaires nettoyés et dédoublonnés ;
   - `bridge.parquet` : la **table de pont**, qui relie chaque joueur-saison à son salaire et
     porte des colonnes dérivées calculées une fois pour toutes : l'**âge**, le **bilan
     victoires** de l'équipe, et l'**ancienneté** dans l'équipe (`team_tenure`, le nombre
     d'années consécutives passées dans la même équipe).

---

## 5. Les serveurs MCP et leurs outils

Deux serveurs (`mcp_servers/`), un par source de données. Tous leurs outils acceptent une
**plage de saisons** (`season_from` / `season_to`, par défaut toute la période), renvoient des
résultats **déjà triés**, dans un **format compact** `{columns, rows}` (les noms de colonnes ne
sont écrits qu'une fois — bien plus léger qu'une liste de dictionnaires, et plus facile à
traiter pour un petit modèle local).

### Serveur `nba-stats` — statistiques de jeu

| Outil | Rôle |
| --- | --- |
| `get_player_season_stats(player, …)` | Stats d'un joueur, une ligne par saison |
| `get_league_leaders(stat, …)` | Meilleurs joueurs pour une statistique sur une plage |
| `search_players(query)` | Recherche de joueurs par nom |
| `list_seasons()` | Saisons disponibles et bornes de la période |

### Serveur `nba-salaries` — salaires et rentabilité

| Outil | Rôle |
| --- | --- |
| **`get_value_ranking(metric, order, …)`** ⭐ | **Classement de rentabilité** : croise stats et salaires (via la table de pont) et calcule, par joueur, le ratio `metric / (salaire / 1 M$)`. `order='best'` → les *bargains*, `order='worst'` → les surpayés. C'est l'outil central du projet. |
| `query_salaries(…)` | Les joueurs les mieux payés sur une plage |
| `get_player_salary(player, …)` | Salaire d'un joueur, une ligne par saison |
| `get_team_payroll(team, season)` | Masse salariale d'une équipe |
| `list_seasons()` | Saisons couvertes par les salaires |

Chaque outil de classement est paramétrable : le paramètre `limit` plafonne le nombre de
**joueurs** retournés (toutes leurs saisons étant conservées), et un mode `aggregate` permet de
renvoyer soit le détail saison par saison (idéal pour visualiser), soit un résumé moyenné par
joueur.

---

## 6. L'agent conversationnel

L'agent est un agent **ReAct** construit avec LangGraph, piloté par un LLM local servi par
Ollama (`qwen3:8b` par défaut). Son fonctionnement :

1. il reçoit la question en français ;
2. il **raisonne** : quel sous-problème, quel outil ?
3. il **appelle l'outil** MCP adéquat avec ses paramètres (métrique, période…) ;
4. il **observe** le résultat (toujours trié, la première ligne est la réponse) ;
5. il **formule** la réponse, chiffres réels à l'appui.

Plusieurs garde-fous ont été mis en place pour fiabiliser un petit modèle local :

- **Aucun chiffre inventé** : toute valeur doit provenir d'un appel d'outil. Le *prompt système*
  l'interdit explicitement et impose des choix par défaut (métrique, période) plutôt que de
  demander des précisions à l'utilisateur.
- **Réponses en français**, concises, annonçant la métrique et la période retenues.
- **Transparence** : un panneau « Étapes de raisonnement » (repliable) affiche en direct chaque
  appel d'outil et son résultat, rendu sous forme de tableau lisible.

**Le chat pilote le tableau de bord.** Après chaque réponse, les filtres du dashboard se mettent
à jour automatiquement en fonction de ce que l'agent a réellement consulté — déduit de ses
**appels d'outils** (source fiable), et non du texte de la réponse. Seuls les filtres pertinents
sont modifiés : la métrique et la période viennent des arguments d'outil, le joueur ou l'équipe
mis en avant viennent du résultat. Ainsi, demander « le joueur le plus rentable en 2021-22 »
sélectionne automatiquement ce joueur, cette métrique et cette saison dans les visualisations.

Exemples de questions : *« Quel est le joueur le plus rentable ? »*, *« Quels sont les joueurs
les plus surpayés ? »*, *« Les 10 mieux payés en 2018-19 ? »*, *« Évolution du salaire de Stephen
Curry »*, *« Quelle équipe a la plus grosse masse salariale en 2021-22 ? »*.

---

## 7. Le tableau de bord

Quatre visualisations (Plotly) s'affichent au-dessus du chat. Les filtres communs — période
(saison de début / de fin), métrique, et sélection cherchable d'équipes et de joueurs — sont
regroupés dans la **barre latérale**.

### V1 — Salaire × performance *(Q1, Q5)*

Un **nuage de points**, un point par joueur (valeurs moyennées sur la période). En abscisse le
salaire (ajusté à l'inflation), en ordonnée la métrique choisie. Une **droite de régression de
référence**, calculée une seule fois sur l'ensemble du dataset, matérialise le rapport
performance/salaire « attendu ». La lecture est immédiate : un point **au-dessus** de la droite
est une bonne affaire, **en dessous** un contrat surpayé. *(Exemple 2021-22 : Brandon Williams,
12,9 pts pour 0,18 M$, soit 71 pts par million — le meilleur rapport de la saison.)*

### V2 — Masse salariale × victoires *(Q2)*

Un point par équipe (moyenne sur la période), masse salariale en abscisse, victoires en
ordonnée, avec une **ligne de tendance**. Elle répond directement à « le budget achète-t-il des
victoires ? » : la corrélation est positive mais modérée, ce qui montre que certaines équipes
sur-performent leur budget et d'autres le gaspillent.

### V3 — Performance selon l'ancienneté *(Q4)*

Un diagramme en barres de la performance moyenne selon le nombre d'années consécutives dans la
même équipe (notre approximation de la « durée de contrat »). La couleur indique le salaire
moyen, ce qui permet de visualiser si l'ancienneté s'accompagne d'un meilleur rendement ou
seulement d'un salaire plus élevé.

### V4 — Rentabilité selon l'âge *(Q6)*

Une courbe de la rentabilité moyenne (performance par million de dollars) selon l'âge. Le **pic**
révèle l'âge où le rapport qualité/prix est le meilleur — typiquement les jeunes joueurs encore
en contrat rookie, autour de 24 ans.

---

## 8. Installation

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
> il est téléchargeable depuis le lien de la section *Données*.

---

## 9. Utilisation

```bash
streamlit run app.py                  # le modèle se décharge après 5 min d'inactivité
streamlit run app.py -- --keep-alive  # garde le modèle en mémoire (pratique en démo)
```

> Le `--` isolé est requis : Streamlit ne transmet au script que ce qui suit ce séparateur.
> En alternative : `KEEP_ALIVE=1 streamlit run app.py`.

Au lancement, l'application **précharge le modèle** en mémoire (la barre latérale affiche
« Modèle prêt ✅ ») pour que la première question soit rapide. Le modèle vit dans le service
Ollama, indépendant de l'application : avec `--keep-alive`, il survit même à un `Ctrl+C` de
Streamlit (à libérer ensuite avec `ollama stop qwen3:8b`).

---

## 10. Structure du projet

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

## 11. Difficultés rencontrées et limites

**La jointure par nom** a été la difficulté technique principale. Sans identifiant commun entre
les deux sources, tout repose sur la correspondance des noms. La normalisation + la table
d'alias atteignent 94,7 % d'appariement ; le choix d'écarter les non-appariés (plutôt que de les
deviner par *fuzzy matching*, qui créait des erreurs) garantit une base fiable, au prix de
quelques joueurs perdus.

**Le pilotage d'un petit LLM local** demande de la rigueur : un modèle de cette taille peut
enchaîner les mauvais outils ou inventer. Les contre-mesures (prompt strict, sorties d'outils
pré-triées et compactes, première ligne = réponse) rendent le comportement fiable, mais restent
des garde-fous plutôt qu'une garantie absolue.

**La « durée de contrat » est une approximation.** Faute de données contractuelles, elle est
estimée par le nombre d'années consécutives d'un joueur dans la même équipe — ce qu'un
spectateur appelle « depuis combien de temps il est là », et non la durée exacte d'un contrat
signé.

**Certaines analyses sont hors périmètre.** L'analyse par poste (Q3) n'est pas couverte : le
poste n'est pas fourni par les endpoints `nba_api` utilisés, et l'obtenir demanderait une
extraction supplémentaire conséquente. Plus fondamentalement, **la performance ne capture pas
tout** : défense, leadership, valeur marketing n'apparaissent pas dans les *box scores*. Un
contrat statistiquement « surpayé » peut donc être justifié par ailleurs — les conclusions de
l'application sont une aide à la décision, pas un verdict.

---

## 12. Pile technique

**Python** · **Streamlit** (interface) · **Plotly** (visualisations) · **Pandas / NumPy**
(traitement) · **LangChain / LangGraph** (agent ReAct) · **Model Context Protocol / FastMCP**
(serveurs d'outils) · **Ollama** (LLM local) · **`nba_api`** (extraction) · **statsmodels**
(régression).
