
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional
from agents.graph import run_query as agent_run
from agents.live_data_agent import LiveDataAgent
from utils.db_connection import test_connection

app = FastAPI(
    title="Hyundai Automobile Agentic AI",
    description=(
        "Agentic AI system for Hyundai queries. "
        "Routes informational queries to RAG Agent and operational queries to Live Data Agent."
    ),
    version="1.0.0",
)


live_agent = LiveDataAgent()




class QueryRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {"query": "Is Hyundai Creta available in Bangalore?"}
        }

class QueryResponse(BaseModel):
    query:          str
    intent:         str
    final_response: str
    error:          Optional[str] = ""



@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    db_ok = test_connection()
    return {
        "status":   "running",
        "service":  "Hyundai Agentic AI",
        "db_status": "connected" if db_ok else "disconnected",
    }


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
def handle_query(request: QueryRequest):
    """
    Main endpoint — accepts any Hyundai-related question.
    The Manager Agent automatically routes it to:
    - RAG Agent   → informational queries (history, specs, features)
    - Live Agent  → operational queries  (availability, pricing, stock)
    - Both        → mixed queries
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = agent_run(request.query)
    return QueryResponse(**result)


@app.get("/inventory", tags=["Live Data"])
def get_inventory_by_city(city: str = Query(..., description="City name e.g. Bangalore")):
    """Get all available Hyundai vehicles in a given city."""
    try:
        data = live_agent.get_inventory_by_city(city)
        return {"city": city, "count": len(data), "inventory": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/{model}", tags=["Live Data"])
def get_model_inventory(
    model: str,
    city: Optional[str] = Query(None, description="Filter by city"),
):
    """Get inventory for a specific Hyundai model, optionally filtered by city."""
    try:
        if city:
            data = live_agent.get_inventory_by_model_and_city(model, city)
        else:
            data = live_agent.get_inventory_by_city(model)   # broad search
        return {"model": model, "city": city, "count": len(data), "inventory": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["Live Data"])
def list_all_models():
    """List all Hyundai models available in the database."""
    try:
        data = live_agent.get_all_models()
        return {"count": len(data), "models": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dealers", tags=["Live Data"])
def get_dealers(city: str = Query(..., description="City name")):
    """Get all Hyundai dealers in a given city."""
    try:
        data = live_agent.get_dealers_by_city(city)
        return {"city": city, "count": len(data), "dealers": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
