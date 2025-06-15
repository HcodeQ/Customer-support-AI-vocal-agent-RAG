from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

loader = PyPDFLoader("mobilax_data.pdf")
documents = loader.load()

#nettoyage du texte : suppression des espaces inutiles
cleaned_text = re.sub(r"\s+", " ", documents[0].page_content.strip())

#chunking du document
chunksize = 200
chunk_overlap = 20
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_text(cleaned_text)

print(chunks)