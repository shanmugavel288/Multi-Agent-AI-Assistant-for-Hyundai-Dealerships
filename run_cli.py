
from agents.graph import run_query

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         HYUNDAI AUTOMOBILE AGENTIC AI ASSISTANT          ║
║  Powered by LangGraph + RAG Agent + Live Data Agent      ║
╠══════════════════════════════════════════════════════════╣
║  Type your question and press Enter.                     ║
║  Type 'exit' or 'quit' to stop.                          ║
╚══════════════════════════════════════════════════════════╝
"""

SAMPLE_QUERIES = [
    "Who founded Hyundai?",
    "What is the price of Hyundai Creta?",
    "Is Hyundai i20 available in Chennai?",
    "Tell me about Creta Electric and its availability in Bangalore.",
    "What ADAS features does Hyundai SmartSense have?",
    "Who is the CEO of Hyundai?",
    "What is the range of Hyundai Ioniq 5?",
]


def main():
    print(BANNER)
    print("Sample queries you can try:")
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Exiting] Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("[Exiting] Goodbye!")
            break

        print("\n" + "─" * 58)
        result = run_query(query)
        print(f"[Intent Detected] {result['intent'].upper()}")
        print("─" * 58)
        print(f"Assistant:\n{result['final_response']}")
        if result.get("error"):
            print(f"\n[Warning] {result['error']}")
        print("─" * 58 + "\n")


if __name__ == "__main__":
    main()
