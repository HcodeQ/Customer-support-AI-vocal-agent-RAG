import logging
import asyncio
from dotenv import load_dotenv
from load_data import vector_store #importer une instance la bd vectorielle
from livekit.agents import BackgroundAudioPlayer, AudioConfig, BuiltinAudioClip
from livekit.agents import ChatMessage
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    ChatContext,
    RunContext,
    function_tool,
    RoomInputOptions,
    Agent,
    AgentSession
)
from livekit.plugins import deepgram
from livekit.plugins import google, silero, noise_cancellation
#from livekit.plugins.turn_detector.multilingual import MutilingualModel


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
        self.vector_store = vector_store

    #récupérer du contexte dans la bd a chaque fois que l'user fini de parler
    async def on_user_turn_completed(
            self, turn_ctx: ChatContext, new_message: ChatMessage,
           ) -> None:
            rag_content = await self.vector_store.similarity_search(new_message.text_content, k=3)
            turn_ctx.add_message(
                role="assistant", 
                content=f"Informations supplémentaires pertinentes pour le prochain message de l'utilisateur: {rag_content}"
            )
            await self.update_chat_ctx(turn_ctx) #mettre à jour l'historique de discussion

    #outil pour récupérer automatique du contexte si l'agent le décide 
    @function_tool
    async def livekit_docs_search(self, context: RunContext, query: str) -> str:
              #avertir l'utilisateur si la recherche prends plus de 1 seconde
              async def _speak_status_update(delay: int = 1):  
                await asyncio.sleep(delay)
                await context.session.generate_reply(instructions=f"""
                   Tu es en train de chercher dans la base de connaissances pour \"{query}\", mais cela prend un peu de temps.
                   Tiens l'utilisateur au courant de ta progression, mais reste très bref.
                """)
              
              try: 
                # Lancer la mise à jour vocale après un court délai
                status_update_task = asyncio.create_task(_speak_status_update(1))
                # Rechercher les documents les plus pertinents
                results = await self.vector_store.similarity_search(query, k=3)
                #arrêter la mise à jour vocale si la recherche est terminée
                status_update_task.cancel()
                try:
                    await status_update_task
                except asyncio.CancelledError:
                    pass
                #logging
                logger.info(
                    f"Results for query: {query}, full context: {results}"
                )
                # Retourner les résultats
                return "".join([result.page_content for result in results])
               
              except Exception as e:
                logger.error(f"Erreur lors de la recherche: {e}")
                return "Une erreur est survenue lors de la recherche."

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        stt=deepgram.STT(),
        llm= google.LLM(model="gemini-2.5-flash-preview-native-audio-dialog"),
        tts= google.TTS(
            gender="male",
            voice_name="fr-FR-Chirp3-HD-Achird"
        ),
        #turn_detection=MutilingualModel(),
        vad=silero.VAD.load()
    )

    await session.start(
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
        agent= RAGEnrichedAgent(),
    )

    background_audio = BackgroundAudioPlayer(
        thinking_sound=[
        AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
        AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
    ],
)
    await background_audio.start(room=ctx.room, agent_session=session)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))