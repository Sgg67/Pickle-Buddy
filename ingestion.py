# load all necessary dependencies
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_tavily import TavilyCrawl

# load env variables
load_dotenv()

# intialize embeddings model to set output to 1024 so it matches Pinecone
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    output_dimensionality=1024
)

# Initialize the vectorstore
vectorstore = PineconeVectorStore(
    index_name="pickleball-docs", 
    embedding=embeddings,
    pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)

# split documents into batches
def create_batches(documents: List[Document], batch_size: int = 50) -> List[List[Document]]:
    """Split a list of documents into batches."""
    return [documents[i: i + batch_size] for i in range(0, len(documents), batch_size)]

# upload batch to pinecone
async def upload_batch(batch: List[Document], batch_num: int) -> bool:
    """Upload a single batch asynchronously to Pinecone."""
    try:
        await vectorstore.aadd_documents(batch)
        print(f"Batch {batch_num} successfully uploaded.")
        return True
    except Exception as e:
        print(f"Failed to upload batch {batch_num}: {e}")
        return False

async def main():
    # urls for the articles on pickleball
    urls = [
        "https://pickleballkitchen.com/14-effective-shots-use-pickleball/",
        "https://www.pickleballmagazine.com/cover-story/25-great-places-to-play-pickleball",
        "https://www.11pickles.com/post/best-pickleball-players",
        "https://thekitchenpickle.com/blogs/gear/best-pickleball-paddles-available/",
    ]

    # get the path of rulebook
    cwd = Path.cwd() / "USAP-Official-Rulebook.pdf"
    # load the pickleball rule book as a doc
    loader = UnstructuredLoader(file_path=str(cwd), chunking_strategy="basic", max_characters=10000000)

    # load the urls as documents
    docs = [UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=1000000).load() for url in urls]

    # add the pickleball rule book into the docs
    docs.append(loader.load())
    # turn the docs into into a list
    docs_list = [item for sublist in docs for item in sublist]
    
    # split characters into tokens
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=200
    )

    # split documents
    doc_splits = text_splitter.split_documents(docs_list)

    # loop through the split docs and clean metadata source
    for doc in doc_splits:
        raw_source = doc.metadata.get("source") or doc.metadata.get("url") or "unknown"
        
        # Check if the source is a local file path (e.g., contains the PDF name)
        if "USAP-Official-Rulebook" in raw_source or str(cwd) in raw_source:
            source = "USAP-Official-Rulebook"
        else:
            source = raw_source
            
        doc.metadata = {"source": source}

    # create batches to be sent to the pinecone database
    batches = create_batches(doc_splits, batch_size=50)

    # sequentially upload batches to pinecone
    successful = 0
    for i, batch in enumerate(batches, start=1):
        if await upload_batch(batch, i):
            successful += 1

if __name__ == "__main__":
    asyncio.run(main())