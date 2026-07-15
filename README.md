# Hyundai Automobile Agentic AI System

## Architecture

```
User Query
    │
    ▼
Manager Agent  (Intent Classification → LangGraph Router)
    │
    ├──► [informational] ──► RAG Agent ──────────────────► Response
    │                         (ChromaDB + LLM)
    │
    ├──► [operational]   ──► Live Data Agent ────────────► Response
    │                         (PostgreSQL + LLM)
    │
    └──► [mixed]         ──► RAG Agent + Live Data Agent ► Merged Response
```

## Tech Stack

| Layer        | Technology                              |
|--------------|-----------------------------------------|
| LLM          | Groq (LLaMA 3 70B) OR Google Gemini 1.5 |
| Orchestration| LangGraph                               |
| RAG Pipeline | LangChain + ChromaDB + HuggingFace      |
| Database     | PostgreSQL (PgAdmin)                    |
| API          | FastAPI + Uvicorn                       |
| Embeddings   | sentence-transformers/all-MiniLM-L6-v2  |

## Project Structure

```
hyundai_ai/
├── .env                        # API keys and config
├── requirements.txt
├── main.py                     # FastAPI app
├── run_cli.py                  # CLI interface
│
├── config/
│   ├── settings.py             # Centralised env loader
│   └── llm_factory.py          # Returns Groq or Gemini LLM
│
├── agents/
│   ├── manager_agent.py        # Intent classifier + router + aggregator
│   ├── rag_agent.py            # ChromaDB retriever + LLM answer
│   ├── live_data_agent.py      # SQL generator + PostgreSQL executor
│   └── graph.py                # LangGraph pipeline
│
├── utils/
│   ├── rag_ingestor.py         # PDF → ChromaDB ingestion script
│   └── db_connection.py        # SQLAlchemy engine + query runner
│
├── db/
│   └── schema.sql              # PostgreSQL schema + full seed data
│
└── data/
    └── Hyundai_RAG_Knowledge_Base.pdf   ← Place your PDF here
```

## Setup Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Configure environment
Edit `.env` with your credentials:
```
LLM_PROVIDER=groq                    # or gemini
GROQ_API_KEY=your_key_here
DB_PASSWORD=your_postgres_password
```

### Step 3 — Set up PostgreSQL database
Open PgAdmin, create a database named `hyundai_db`, then run:
```bash
psql -U postgres -d hyundai_db -f db/schema.sql
```

### Step 4 — Ingest the PDF into ChromaDB
Place `Hyundai_RAG_Knowledge_Base.pdf` in the `./data/` folder, then run:
```bash
python -m utils.rag_ingestor
```
This only needs to be done once.

### Step 5 — Run the system

**Option A — Interactive CLI:**
```bash
python run_cli.py
```

**Option B — REST API (FastAPI):**
```bash
uvicorn main:app --reload --port 8000
```
Then open: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint              | Description                              |
|--------|-----------------------|------------------------------------------|
| GET    | `/`                   | Health check                             |
| POST   | `/query`              | Main agent endpoint — any question       |
| GET    | `/inventory?city=`    | All available vehicles in a city         |
| GET    | `/inventory/{model}`  | Inventory for a specific model           |
| GET    | `/models`             | List all models in database              |
| GET    | `/dealers?city=`      | All dealers in a city                    |

## Example Queries

| Query                                              | Intent        | Route         |
|----------------------------------------------------|---------------|---------------|
| "Who founded Hyundai?"                             | informational | RAG Agent     |
| "What is Hyundai's EV strategy?"                   | informational | RAG Agent     |
| "Is Creta available in Bangalore?"                 | operational   | Live Data Agent|
| "What is the price of i20 in Chennai?"             | operational   | Live Data Agent|
| "Tell me about Creta and its availability in Delhi"| mixed         | Both Agents   |

## API Request Example (POST /query)
```json
{
  "query": "Is Hyundai Creta available in Bangalore?"
}
```

Response:
```json
{
  "query": "Is Hyundai Creta available in Bangalore?",
  "intent": "operational",
  "final_response": "Yes! Hyundai Creta is available in Bangalore...",
  "error": ""
}
```
