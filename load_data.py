from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_together import TogetherEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

loader = PDFMinerLoader("mobilax_data.pdf")
documents = loader.load()

#nettoyage du texte : suppression des espaces inutiles
cleaned_text = re.sub(r"\s+", " ", documents[0].page_content.strip())

#chunking du document
chunksize = 200
chunk_overlap = 20
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_text(cleaned_text)

# Chargement de la clé API Together depuis le fichier .env
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

#initialisation du modèle d'embeddings
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval", api_key=api_key)

#initialisation de la base de données vectorielle
vector_store = Chroma(
    collection_name="sav_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

#ajout des documents à la base de données vectorielle avec add_texts car pas besoin de metadata
vector_store.add_texts(chunks)