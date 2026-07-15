


from typing import TypedDict, Literal
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.llm_factory import get_llm



class AgentState(TypedDict):
    query: str                          
    intent: str                         
    rag_response: str                   
    live_response: str               
    final_response: str                 
    error: str                          



INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Manager Agent for a Hyundai Automobile Assistant.
Your only job is to classify the user's query into exactly one of these categories:

  - informational : Questions about Hyundai history, founders, CEO, company facts,
                    car model specs/features, awards, technology, EV strategy, etc.
  - operational   : Questions about real-time data — vehicle availability, stock,
                    pricing, waiting periods, dealer locations, service centres.
  - mixed         : The query asks BOTH informational AND operational things at once.

Reply with exactly one word: informational, operational, or mixed.
No explanation. No punctuation. Just the single word."""),
    ("human", "{query}"),
])


# ── Response Aggregation Prompt ────────────────────────────────
AGGREGATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Hyundai Automobile Assistant.
Combine the following two responses into a single, coherent, friendly answer.
Do not repeat information. Be concise and structured.

Informational response:
{rag_response}

Operational response:
{live_response}
"""),
    ("human", "User's original question: {query}"),
])


class ManagerAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.intent_chain = INTENT_PROMPT | self.llm | StrOutputParser()
        self.aggregate_chain = AGGREGATE_PROMPT | self.llm | StrOutputParser()

   
    def classify_intent(self, state: AgentState) -> AgentState:
        print(f"\n[Manager] Classifying intent for: '{state['query']}'")
        try:
            intent = self.intent_chain.invoke({"query": state["query"]}).strip().lower()
          
            if intent not in ("informational", "operational", "mixed"):
                intent = "informational"
            print(f"[Manager] Intent → {intent}")
            return {**state, "intent": intent}
        except Exception as e:
            return {**state, "intent": "informational", "error": str(e)}

   
    def route(self, state: AgentState) -> Literal["rag_agent", "live_agent", "both"]:
        intent = state.get("intent", "informational")
        if intent == "operational":
            return "live_agent"
        elif intent == "mixed":
            return "both"
        return "rag_agent"


    def aggregate_responses(self, state: AgentState) -> AgentState:
        print("[Manager] Aggregating RAG + Live responses.")
        try:
            merged = self.aggregate_chain.invoke({
                "rag_response":  state.get("rag_response", ""),
                "live_response": state.get("live_response", ""),
                "query":         state["query"],
            })
            return {**state, "final_response": merged}
        except Exception as e:
            combined = (
                state.get("rag_response", "") + "\n\n" + state.get("live_response", "")
            ).strip()
            return {**state, "final_response": combined, "error": str(e)}

    def finalise(self, state: AgentState) -> AgentState:
        if state.get("rag_response") and not state.get("final_response"):
            return {**state, "final_response": state["rag_response"]}
        if state.get("live_response") and not state.get("final_response"):
            return {**state, "final_response": state["live_response"]}
        return state
