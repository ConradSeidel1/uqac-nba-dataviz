"""App Streamlit — Assistant NBA (agent ReAct + LLM local Ollama + serveurs MCP).

Un chat où un agent ReAct (LangChain / LangGraph) tourne sur un LLM local via
Ollama et interroge les données du projet à travers les deux serveurs MCP :
  • nba-stats     → statistiques de jeu (data/clean/stats_clean.parquet)
  • nba-salaries  → salaires (data/clean/salaries_clean.parquet + bridge.parquet)

Aucune dépendance à Claude Desktop : le LLM est local, les serveurs MCP sont lancés
en sous-processus (transport stdio) par l'app elle-même.

Prérequis :
    pip install -r requirements.txt
    ollama pull qwen3:8b
    python build_clean_datasets.py

Lancer :
    streamlit run app.py                  # modèle déchargé après 5 min d'inactivité
    streamlit run app.py -- --keep-alive  # modèle gardé en mémoire (utile en démo)
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import streamlit as st

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
SERVERS_DIR = ROOT / "mcp_servers"
DEFAULT_MODEL = "qwen3:8b"

# Durée pendant laquelle Ollama garde le modèle en VRAM.
#   keep-alive activé  -> -1 : gardé indéfiniment (à libérer avec `ollama stop <modèle>`).
#   sinon              -> "5m" : déchargé après 5 min d'inactivité (défaut).
# Activer : soit `streamlit run app.py -- --keep-alive` (le « -- » est obligatoire),
# soit la variable d'environnement KEEP_ALIVE=1.
_KEEP_FLAGS = {"--keep-alive", "--keepalive", "--keep_alive"}
_keep_alive_on = (
    any(a in _KEEP_FLAGS for a in sys.argv)
    or os.environ.get("KEEP_ALIVE", "").lower() in {"1", "true", "yes", "on"}
)
KEEP_ALIVE = -1 if _keep_alive_on else "5m"

SYSTEM_PROMPT = (
    "Tu es un assistant d'analyse NBA pour un *front office* d'équipe.\n\n"
    "LANGUE : tu réponds TOUJOURS en FRANÇAIS, quelle que soit la langue des données "
    "ou de la question. Jamais d'anglais dans ta réponse.\n\n"
    "RÈGLES IMPÉRATIVES :\n"
    "1. Tu DOIS utiliser les outils MCP (stats et salaires) pour obtenir des chiffres. "
    "N'invente JAMAIS de nom de joueur, de statistique ou de salaire. Tout chiffre que "
    "tu donnes doit provenir d'un appel d'outil que tu viens d'effectuer.\n"
    "2. Les outils renvoient déjà les résultats TRIÉS dans le bon ordre : la PREMIÈRE "
    "ligne de la réponse est la réponse à la question (le meilleur / le plus rentable / "
    "le mieux payé, selon le tri). Pour « le joueur le plus X », prends la 1re ligne, "
    "ne parcours pas toute la liste et n'inventes pas d'analyse sur les autres lignes.\n"
    "3. Tu ne demandes JAMAIS de précision et tu ne proposes pas plusieurs méthodes. "
    "Face à une question vague, tu CHOISIS une métrique et une période, tu l'annonces "
    "en une phrase, puis tu réponds.\n"
    "4. Choix par défaut : saison « 2021-22 » ; rentabilité = get_value_ranking avec "
    "metric='pts', order='best' (meilleur rapport points/salaire). Pour « surpayé », "
    "order='worst'. Demande peu de lignes (limit=5 à 10) quand tu veux un classement.\n"
    "5. Réponse concise : annonce la métrique et la période choisies, donne le ou les "
    "joueurs en tête avec leurs chiffres exacts, puis une phrase de conclusion.\n\n"
    "Les saisons sont au format '2018-19'."
)

# Définition des deux serveurs MCP, lancés en stdio par l'app.
MCP_SERVERS = {
    "nba-stats": {
        "command": sys.executable,
        "args": [str(SERVERS_DIR / "nba_stats_server.py")],
        "transport": "stdio",
    },
    "nba-salaries": {
        "command": sys.executable,
        "args": [str(SERVERS_DIR / "nba_salaries_server.py")],
        "transport": "stdio",
    },
}


# --------------------------------------------------------------------------- #
# Agent (async) — construit une fois, mis en cache
# --------------------------------------------------------------------------- #

async def _build_agent(model_name: str):
    """Connecte les serveurs MCP, récupère les outils, crée l'agent ReAct."""
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    # keep_alive : le modèle vit dans le service Ollama (pas dans ce process) :
    # il survit à un Ctrl+C de Streamlit.
    llm = ChatOllama(model=model_name, temperature=0, keep_alive=KEEP_ALIVE)
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent, tools, llm


async def _warmup(llm) -> None:
    """Force Ollama à charger le modèle en VRAM (évite le cold start à la 1re question)."""
    await llm.ainvoke("ping")


def get_agent(model_name: str):
    """Construit l'agent (et sa boucle asyncio dédiée) une seule fois par session."""
    if "agent" not in st.session_state or st.session_state.get("model") != model_name:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agent, tools, llm = loop.run_until_complete(_build_agent(model_name))
        st.session_state.loop = loop
        st.session_state.agent = agent
        st.session_state.tools = tools
        st.session_state.llm = llm
        st.session_state.model = model_name
        st.session_state.warmed_up = False
    return st.session_state.agent, st.session_state.loop


def warmup_model() -> None:
    """Charge le modèle en VRAM une seule fois par session (idempotent)."""
    if not st.session_state.get("warmed_up", False):
        st.session_state.loop.run_until_complete(_warmup(st.session_state.llm))
        st.session_state.warmed_up = True


def _format_tool_result(obs) -> str:
    """Rend lisible un résultat d'outil MCP dans le panneau « étapes ».

    Le contenu MCP arrive souvent enveloppé : [{'type': 'text', 'text': '<json>'}].
    On déballe le texte, on parse le JSON, et si c'est notre format compact
    {columns, rows} on l'affiche en petit tableau Markdown. Sinon, on affiche le
    JSON de façon compacte (une ligne).
    """
    import json

    # 1) Déballer l'enveloppe MCP -> texte brut.
    raw = obs
    if isinstance(obs, list) and obs and isinstance(obs[0], dict) and "text" in obs[0]:
        raw = obs[0]["text"]
    elif isinstance(obs, dict) and "text" in obs:
        raw = obs["text"]

    # 2) Tenter de parser le JSON.
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return f"**◀ Résultat** : {str(raw)[:400]}"

    # 3) Format compact {columns, rows} -> tableau Markdown (max ~8 lignes affichées).
    if isinstance(data, dict) and "columns" in data and "rows" in data:
        cols = data["columns"]
        rows = data["rows"][:8]
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
        note = ""
        if data.get("truncated"):
            note = (f"\n\n_↳ {data.get('players_returned')} joueurs affichés sur "
                    f"{data.get('total_players')} (tronqué)._")
        more = f"\n\n_… {data['n_rows'] - len(rows)} lignes de plus_" \
            if data.get("n_rows", 0) > len(rows) else ""
        return f"**◀ Résultat**\n\n{header}\n{sep}\n{body}{more}{note}"

    # 4) Autre dict (erreur, payroll…) -> JSON compact une ligne.
    return f"**◀ Résultat** : {json.dumps(data, ensure_ascii=False)[:400]}"


async def _stream_steps(agent, question: str, steps_box) -> str:
    """Streame les étapes de l'agent (Thought / Action / Observation).

    Affiche au fil de l'eau, dans `steps_box`, la réflexion de l'agent, ses appels
    d'outils et leurs résultats ; renvoie la réponse finale. Reprend la logique de
    `show_steps` du cours sur `agent.astream(...)`.
    """
    lines: list[str] = []
    final_answer = ""
    used_a_tool = False

    async for chunk in agent.astream({"messages": [("user", question)]}):
        if "agent" in chunk:
            msg = chunk["agent"]["messages"][0]
            if getattr(msg, "tool_calls", None):
                if msg.content:
                    thought = str(msg.content).replace("\n", " ")[:300]
                    lines.append(f"**💭 Réflexion** : {thought}")
                for tc in msg.tool_calls:
                    used_a_tool = True
                    lines.append(f"**▶ Outil** `{tc['name']}` → `{tc['args']}`")
            elif msg.content:
                final_answer = msg.content
        elif "tools" in chunk:
            obs = chunk["tools"]["messages"][0].content
            lines.append(_format_tool_result(obs))
        if lines:
            steps_box.markdown("\n\n".join(lines))

    if not used_a_tool:
        lines.append(
            "_⚠️ L'agent a répondu sans appeler d'outil MCP — la réponse n'est donc "
            "pas appuyée sur les données. Reformule ou relance._")
        steps_box.markdown("\n\n".join(lines))

    st.session_state["_last_steps"] = "\n\n".join(lines)
    return final_answer


def ask_agent(question: str, steps_box) -> str:
    """Pose une question à l'agent en streamant ses étapes ; renvoie la réponse finale."""
    loop = st.session_state.loop
    return loop.run_until_complete(_stream_steps(st.session_state.agent, question, steps_box))


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Assistant NBA", page_icon="🏀", layout="centered")
    st.title("🏀 Assistant NBA — Performance vs. Contrats")
    st.caption("Agent ReAct sur LLM local (Ollama) + serveurs MCP nba-stats & nba-salaries")

    with st.sidebar:
        st.header("Configuration")
        model_name = st.text_input("Modèle Ollama", value=DEFAULT_MODEL)
        if KEEP_ALIVE == -1:
            st.caption("🔒 Modèle gardé en mémoire (keep-alive actif)")
        else:
            st.caption("⏳ Modèle déchargé après 5 min d'inactivité")
            with st.expander("Activer le keep-alive ?"):
                st.markdown(
                    "Lance avec **l'une** de ces commandes :\n\n"
                    "```\nstreamlit run app.py -- --keep-alive\n```\n"
                    "ou (fallback) :\n\n"
                    "```\nKEEP_ALIVE=1 streamlit run app.py\n```\n"
                    f"_Args reçus : `{list(sys.argv[1:])}`_")
        show_reasoning = st.checkbox("Afficher les étapes de raisonnement", value=False)
        if st.button("Réinitialiser la conversation"):
            st.session_state.pop("messages", None)
            st.rerun()

    # Historique de conversation. Chaque message = (role, content, steps).
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for role, content, steps in st.session_state.messages:
        with st.chat_message(role):
            if steps:
                with st.expander("🧠 Étapes de raisonnement"):
                    st.markdown(steps)
            st.markdown(content)

    # Construction de l'agent (une fois).
    try:
        get_agent(model_name)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Impossible de démarrer l'agent : {exc}\n\n"
            "Vérifie qu'Ollama tourne, que le modèle est téléchargé, et que "
            "`data/clean/*.parquet` existe.")
        st.stop()

    # Warmup : charge le modèle en VRAM au démarrage pour une 1re question rapide.
    if not st.session_state.get("warmed_up", False):
        with st.spinner(f"Préchargement du modèle {model_name} en mémoire…"):
            try:
                warmup_model()
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Préchargement impossible ({exc}). "
                           "La première question sera plus lente.")
    if st.session_state.get("warmed_up"):
        st.sidebar.success(f"Modèle {model_name} prêt ✅")

    # Saisie utilisateur.
    if question := st.chat_input("Pose ta question (ex. « Qui est le joueur le plus rentable ? »)"):
        st.session_state.messages.append(("user", question, ""))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            steps_text = ""
            with st.expander("🧠 Étapes de raisonnement", expanded=show_reasoning):
                steps_box = st.empty()

            with st.spinner("L'agent interroge les données…"):
                try:
                    answer = ask_agent(question, steps_box)
                    steps_text = st.session_state.get("_last_steps", "")
                except Exception as exc:  # noqa: BLE001
                    answer = f"Erreur pendant le raisonnement de l'agent : {exc}"
            st.markdown(answer)

        st.session_state.messages.append(("assistant", answer, steps_text))


if __name__ == "__main__":
    main()
