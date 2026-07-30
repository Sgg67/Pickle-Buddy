# import necessary dependencies
import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver  


# load env variables
load_dotenv()

# intialize checkpoint saver globally
checkpointer=InMemorySaver()

# Initialize Google GenAI Embeddings
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

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""

    # Retrieve the top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(query)
    

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and retrived doc artifacts
    return serialized, retrieved_docs


# run the rag pipeline to answers users query's with context
def run_llm(query: str, thread_id: str = "default_thread") -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    """

    system_prompt = (
        "You are helpful AI assistant that answers questions about pickleball paddles, professional players, rules, shots, and best cities to play pickleball. "
        "You have access to a tool that retrieves relevant documentation. "
        "Provide detailed, comprehensive explanations with code examples or bullet points where relevant. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the most relevant sources you use in your answers at the end of the response. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    # create an agent and provide it tools
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt, checkpointer=checkpointer)

    # add the configurable thread id
    config = {"configurable": {"thread_id": thread_id}}
    # assign the role as user and the content as the user generated query
    messages = [{"role": "user", "content": query}]


    # call the agent with the given message
    response = agent.invoke({"messages": messages}, config = config)

    # show the ai's last response
    last_message = response["messages"][-1]

    # if it is the last messase
    if hasattr(last_message, "text") and last_message.text:
        # then that is ai's answer
        answer = last_message.text
    else:
        # get the content of the last message
        raw_content = last_message.content
        # if the answer is a string that is the last content
        if isinstance(raw_content, str):
            answer = raw_content
        # otherwise check if the content is a list
        elif isinstance(raw_content, list):
            # get a list of the text parts
            text_parts = []
            # loop through the text parts
            for block in raw_content:
                # if the content is a dictionary
                if isinstance(block, dict):
                    text_parts.append(block.get("text", ""))
                    # if the content is a dictionary
                elif hasattr(block, "text"):
                    text_parts.append(getattr(block, "text", ""))
                    # if the content is a string
                elif isinstance(block, str):
                    text_parts.append(block)
            # add \n plus text_parts as the answer
            answer = "\n".join(filter(None, text_parts))
        else:
            # otherwise the raw content is a string
            answer = str(raw_content)

    # extract context documents from ToolMessage artifacts
    context_docs = []
    # loop through the response messages
    for message in response["messages"]:
        # if the message is a tool and the tagged attribute is artifact
            if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
                if isinstance(message.artifact, list):
                    context_docs.extend(message.artifact)

    # return the answer and the context of the answer
    return {
        "answer": answer,
        "context": context_docs
    }




