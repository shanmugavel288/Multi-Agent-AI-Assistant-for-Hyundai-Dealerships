

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config.llm_factory import get_llm
from utils.rag_ingestor import get_vectorstore
from agents.manager_agent import AgentState


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable Hyundai Automobile Expert.
Answer the user's question using ONLY the context provided below.
Be accurate, helpful, and concise. If the context doesn't contain enough
information, say "I don't have enough information about that in my knowledge base."

Context:
{context}
"""),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    """Join retrieved document chunks into a single context string."""
    return "\n\n".join(
        f"[Source: Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


class RAGAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        print("[RAG Agent] Loading vector store from ChromaDB...")
        self.vectorstore = get_vectorstore()
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},        # top-5 most relevant chunks
        )
        self.chain = (
            {
                "context":  self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        print("[RAG Agent] Ready.")

 
    def run(self, state: AgentState) -> AgentState:
        query = state["query"]
        print(f"[RAG Agent] Processing: '{query}'")
        try:
            answer = self.chain.invoke(query)
            print(f"[RAG Agent] Answer generated ({len(answer)} chars).")
            return {**state, "rag_response": answer}
        except Exception as e:
            error_msg = f"RAG Agent error: {e}"
            print(f"[RAG Agent] ERROR: {e}")
            return {**state, "rag_response": "", "error": error_msg}

    def retrieve_only(self, query: str) -> list[dict]:
        """
        Returns raw retrieved chunks (useful for debugging / inspection).
        """
        docs = self.retriever.invoke(query)
        return [
            {
                "content": d.page_content,
                "page": d.metadata.get("page"),
                "source": d.metadata.get("source"),
            }
            for d in docs
        ]
