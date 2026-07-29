# load all necessary dependencies
import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_unstructured import UnstructuredLoader
# load env variables
load_dotenv()

urls = [
    "https://usapickleball.org/rules/",
    "https://pickleballkitchen.com/14-effective-shots-use-pickleball/",
    "https://happypeoplepickleball.com/best-places-to-play-pickleball/",
    "https://www.11pickles.com/post/best-pickleball-players",
    "https://thekitchenpickle.com/blogs/gear/best-pickleball-paddles-available/",
]

docs = [UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=1000000).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)


# intialize embeddings model to set output to 1024 so it matches Pinecone
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    output_dimensionality=1024
)

doc_splits = text_splitter.split_documents(docs_list)
# Initialize the vectorstore
vectorstore = PineconeVectorStore(
    index_name="docs", 
    embedding=embeddings,
    pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)

retriever = vectorstore.as_retriever()