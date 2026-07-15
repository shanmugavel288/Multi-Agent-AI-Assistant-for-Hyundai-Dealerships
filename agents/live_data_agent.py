# agents/live_data_agent.py
# ─────────────────────────────────────────────────────────────
# Live Data Agent — Operational Layer
# Queries PostgreSQL for real-time inventory, pricing, availability
# ─────────────────────────────────────────────────────────────

import re
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.llm_factory import get_llm
from utils.db_connection import run_query
from agents.manager_agent import AgentState


# ── SQL Generation Prompt ─────────────────────────────────────
SQL_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a PostgreSQL expert for a Hyundai dealership database.
Generate a single valid SQL SELECT query to answer the user's question.

DATABASE SCHEMA:
  models(model_id, model_name, segment, fuel_type, variant, price_min, price_max, seating, launched_year)
  dealers(dealer_id, dealer_name, city, state, address, phone, email)
  inventory(inventory_id, model_id, dealer_id, available_stock, waiting_period, color_options, last_updated)

RULES:
- Use ILIKE for case-insensitive string matching (city, model_name, etc.)
- JOIN inventory with models and dealers when needed
- Always return meaningful columns (model_name, city, available_stock, waiting_period, price_min, price_max, color_options)
- If asking about availability, filter WHERE available_stock > 0
- Output ONLY the raw SQL query. No explanation, no markdown, no backticks.
"""),
    ("human", "{query}"),
])

# ── Natural Language Response Prompt ──────────────────────────
NL_RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Hyundai dealership assistant.
Convert the following database query results into a clear, helpful natural language response.
Include all relevant details: model, city, stock, price range, waiting period, colours.
If results are empty, say the vehicle is currently unavailable in that location.

Query Results:
{results}
"""),
    ("human", "Original question: {query}"),
])


class LiveDataAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.sql_chain   = SQL_GEN_PROMPT | self.llm | StrOutputParser()
        self.nl_chain    = NL_RESPONSE_PROMPT | self.llm | StrOutputParser()
        print("[Live Data Agent] Ready.")

    # ── Step 1: Generate SQL ───────────────────────────────────
    def generate_sql(self, query: str) -> str:
        raw = self.sql_chain.invoke({"query": query}).strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:sql)?|```", "", raw).strip()
        print(f"[Live Data Agent] Generated SQL:\n  {raw}")
        return raw

    # ── Step 2: Execute SQL safely ────────────────────────────
    def execute_sql(self, sql: str) -> list[dict]:
        # Safety: only allow SELECT statements
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted.")
        return run_query(sql)

    # ── Step 3: Format results to natural language ─────────────
    def format_response(self, query: str, results: list[dict]) -> str:
        results_str = str(results) if results else "No records found."
        return self.nl_chain.invoke({"query": query, "results": results_str})

    # ── LangGraph Node ────────────────────────────────────────
    def run(self, state: AgentState) -> AgentState:
        query = state["query"]
        print(f"[Live Data Agent] Processing: '{query}'")
        try:
            sql     = self.generate_sql(query)
            results = self.execute_sql(sql)
            print(f"[Live Data Agent] Got {len(results)} row(s) from DB.")
            answer  = self.format_response(query, results)
            return {**state, "live_response": answer}
        except Exception as e:
            error_msg = f"Live Data Agent error: {e}"
            print(f"[Live Data Agent] ERROR: {e}")
            return {**state, "live_response": "", "error": error_msg}

    # ── Direct query helpers (for FastAPI routes) ─────────────
    def get_inventory_by_city(self, city: str) -> list[dict]:
        sql = """
            SELECT m.model_name, m.variant, m.fuel_type,
                   d.city, d.state, d.dealer_name, d.phone,
                   i.available_stock, i.waiting_period,
                   m.price_min, m.price_max, i.color_options
            FROM inventory i
            JOIN models  m ON i.model_id  = m.model_id
            JOIN dealers d ON i.dealer_id = d.dealer_id
            WHERE d.city ILIKE :city AND i.available_stock > 0
            ORDER BY m.model_name;
        """
        return run_query(sql, {"city": f"%{city}%"})

    def get_inventory_by_model_and_city(self, model: str, city: str) -> list[dict]:
        sql = """
            SELECT m.model_name, m.variant, m.fuel_type,
                   d.city, d.dealer_name, d.phone,
                   i.available_stock, i.waiting_period,
                   m.price_min, m.price_max, i.color_options
            FROM inventory i
            JOIN models  m ON i.model_id  = m.model_id
            JOIN dealers d ON i.dealer_id = d.dealer_id
            WHERE m.model_name ILIKE :model
              AND d.city       ILIKE :city
            ORDER BY i.available_stock DESC;
        """
        return run_query(sql, {"model": f"%{model}%", "city": f"%{city}%"})

    def get_all_models(self) -> list[dict]:
        sql = """
            SELECT DISTINCT model_name, segment, fuel_type,
                            price_min, price_max, seating, launched_year
            FROM models ORDER BY model_name;
        """
        return run_query(sql)

    def get_dealers_by_city(self, city: str) -> list[dict]:
        sql = """
            SELECT dealer_name, city, state, address, phone, email
            FROM dealers
            WHERE city ILIKE :city
            ORDER BY dealer_name;
        """
        return run_query(sql, {"city": f"%{city}%"})
