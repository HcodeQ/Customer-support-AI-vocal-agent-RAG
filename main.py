import logging
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    RunContext,
    function_tool,
    RoomInputOptions,
    Agent,
    AgentSession
)

from livekit.plugins import google, silero, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MutilingualModel

#load environment variables
load_dotenv()

#configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-agent")

class RAGEnrichedAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            Tu es un assistant vocal spécialisé dans le support client pour Mobilax.
            Tu peux répondre aux questions sur les commandes, le support technique,
            les retours et remboursements. Tes réponses doivent toujours être concises 
            et adaptées à la synthèse vocale, donc reste naturel et évite d'utiliser des 
            formatages spéciaux.
            """
        )

