# Pickle Buddy

**Pickle Buddy** is an intelligent, Retrieval-Augmented Generation (RAG) conversational assistant designed to answer all your questions about pickleball! Powered by Google Gemini, LangChain, and Pinecone, Pickle Buddy fetches accurate information on official USAP rules, top equipment, technique/shots, pro players, and popular places to play.

 **Application:** [Pickle Buddy on Streamlit](https://pickle-buddy.streamlit.app/)

---

##  Key Features

- **Retrieval-Augmented Answers (RAG):** Answers queries using grounded data scraped from top pickleball web resources and the official USAP Rulebook.
- **Source Attribution:** Provides explicit source links and document citations alongside generated answers for complete transparency.
- **Stateful Conversational Memory:** Powered by LangGraph's `InMemorySaver`, allowing the AI agent to maintain continuous thread history for multi-turn conversations.
- **Session Management:** Built-in Streamlit sidebar controls to clear chat memory and generate new thread sessions on demand.
- **Async Vector Ingestion Pipeline:** Custom asynchronous ingestion script to chunk, embed, and batch-upload vector data into Pinecone efficiently.

---

## Built With

### **Frontend & Interface**
- **[Streamlit](https://streamlit.io/):** Interactive Python web app UI for real-time chat.

### **AI & RAG Orchestration**
- **[LangChain](https://www.langchain.com/):** Agent workflow creation, custom retrieval tools, and structured data handling.
- **[LangGraph](https://www.langchain.com/langgraph):** Handles thread state persistence (`InMemorySaver`).
- **[Google Gemini AI](https://deepmind.google/technologies/gemini/):**
  - Model: `gemini-2.5-flash` for high-speed agent reasoning and generation.
  - Embeddings: `models/gemini-embedding-001` (configured to 1024 dimensions).

### **Vector DB & Data Processing**
- **[Pinecone](https://www.pinecone.io/):** Serverless vector index storage (`pickleball-docs`).
- **[Unstructured](https://unstructured.io/):** Extracts raw text from local PDF documents (`USAP-Official-Rulebook.pdf`) and web URLs.

---

## Project Architecture

```text
├── app.py                   # Main Streamlit web frontend & chat state management
├── backend/
│   ├── core.py              # LangChain RAG agent implementation and retrieval logic
│   └── ingestion.py         # Document parsing, chunking, embedding, and Pinecone indexing
├── USAP-Official-Rulebook.pdf # Local domain knowledge base document
├── .env                     # Environment secret key configuration
└── requirements.txt         # Project dependencies