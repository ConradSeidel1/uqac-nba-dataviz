"""App Streamlit — Assistant NBA (agent ReAct + LLM local Ollama + serveurs MCP).

Un chat où un agent ReAct (LangChain / LangGraph) tourne sur un LLM local via
Ollama et interroge les données du projet à travers les deux serveurs MCP :
  • nba-stats     → statistiques de jeu (data/clean/stats_clean.parquet)
  • nba-salaries  → salaires (data/clean/salaries_clean.parquet + bridge.parquet)

Aucune dépendance à Claude Desktop : le LLM est local, les serveurs MCP sont lancés
en sous-processus (transport stdio) par l'app elle-même.

Prérequis :
    pip install -r requirements.txt
    ollama pull qwen3:8b             # LLM local (8 Go VRAM)
    python build_clean_datasets.py   # génère data/clean/*.parquet

Lancer :
    streamlit run app.py
"""

from __future__ import annotations

import asyncio
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

SYSTEM_PROMPT = (
    "Tu es un assistant d'analyse NBA pour un *front office* d'équipe. "
    "Tu réponds en français, de façon chiffrée et sourcée. "
    "Tu ne calcules JAMAIS les chiffres toi-même : tu utilises les outils MCP "
    "(stats et salaires) pour récupérer les données, puis tu les commentes. "
    "Les saisons sont au format '2018-19'. Si une question est ambiguë "
    "(ex. « le meilleur joueur »), précise la métrique que tu retiens."
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
    llm = ChatOllama(model=model_name, temperature=0)
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent, tools


def get_agent(model_name: str):
    """Construit l'agent (et sa boucle asyncio dédiée) une seule fois par session."""
    if "agent" not in st.session_state or st.session_state.get("model") != model_name:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agent, tools = loop.run_until_complete(_build_agent(model_name))
        st.session_state.loop = loop
        st.session_state.agent = agent
        st.session_state.tools = tools
        st.session_state.model = model_name
    return st.session_state.agent, st.session_state.loop


def ask_agent(question: str) -> str:
    """Pose une question à l'agent et renvoie la réponse finale en texte."""
    agent = st.session_state.agent
    loop = st.session_state.loop
    result = loop.run_until_complete(
        agent.ainvoke({"messages": [("user", question)]}))
    # La réponse finale est le dernier message de l'agent.
    return result["messages"][-1].content


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
        st.markdown(
            "**Prérequis**\n\n"
            f"- `ollama pull {DEFAULT_MODEL}`\n"
            "- `python build_clean_datasets.py`\n"
        )
        if st.button("Réinitialiser la conversation"):
            st.session_state.pop("messages", None)
            st.rerun()

    # Historique de conversation (affichage).
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    # Construction de l'agent (une fois). Erreurs explicites si Ollama/MCP indispo.
    try:
        get_agent(model_name)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Impossible de démarrer l'agent : {exc}\n\n"
            "Vérifie qu'Ollama tourne, que le modèle est téléchargé "
            f"(`ollama pull {model_name}`), et que `data/clean/*.parquet` existe.")
        st.stop()

    # Saisie utilisateur.
    if question := st.chat_input("Pose ta question (ex. « Qui a le meilleur ratio points/salaire en 2018-19 ? »)"):
        st.session_state.messages.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("L'agent interroge les données…"):
                try:
                    answer = ask_agent(question)
                except Exception as exc:  # noqa: BLE001
                    answer = f"Erreur pendant le raisonnement de l'agent : {exc}"
            st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))


if __name__ == "__main__":
    main()
