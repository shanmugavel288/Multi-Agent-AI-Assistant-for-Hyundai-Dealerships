

from langgraph.graph import StateGraph, END
from agents.manager_agent import ManagerAgent, AgentState
from agents.rag_agent import RAGAgent
from agents.live_data_agent import LiveDataAgent


def build_graph() -> StateGraph:
   
    manager   = ManagerAgent()
    rag       = RAGAgent()
    live      = LiveDataAgent()

    graph = StateGraph(AgentState)

   
    graph.add_node("classify_intent",   manager.classify_intent)
    graph.add_node("rag_node",          rag.run)
    graph.add_node("live_node",         live.run)
    graph.add_node("aggregate",         manager.aggregate_responses)
    graph.add_node("finalise",          manager.finalise)

  
    graph.set_entry_point("classify_intent")

  
    graph.add_conditional_edges(
        "classify_intent",
        manager.route,
        {
            "rag_agent":  "rag_node",
            "live_agent": "live_node",
            "both":       "rag_node",       
        },
    )

 
    def after_rag(state: AgentState):
        return "live_node" if state.get("intent") == "mixed" else "finalise"

    graph.add_conditional_edges("rag_node", after_rag, {
        "live_node": "live_node",
        "finalise":  "finalise",
    })

    def after_live(state: AgentState):
        return "aggregate" if state.get("intent") == "mixed" else "finalise"

    graph.add_conditional_edges("live_node", after_live, {
        "aggregate": "aggregate",
        "finalise":  "finalise",
    })

   
    graph.add_edge("aggregate", END)
    graph.add_edge("finalise",  END)

    return graph.compile()



hyundai_graph = build_graph()


def run_query(user_query: str) -> dict:
    """
    Main entry point. Accepts a user question, runs the full graph,
    returns a dict with intent + final_response.
    """
    initial_state: AgentState = {
        "query":          user_query,
        "intent":         "",
        "rag_response":   "",
        "live_response":  "",
        "final_response": "",
        "error":          "",
    }
    final_state = hyundai_graph.invoke(initial_state)
    return {
        "query":          final_state["query"],
        "intent":         final_state["intent"],
        "final_response": final_state["final_response"],
        "error":          final_state.get("error", ""),
    }
