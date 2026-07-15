# config/llm_factory.py
# ─────────────────────────────────────────────────────────────
# Returns the configured LLM instance (Groq or Gemini)
# ─────────────────────────────────────────────────────────────

from config.settings import settings


def get_llm(temperature: float = 0.0):
    """
    Returns a LangChain-compatible LLM.
    Switch between Groq and Gemini via .env LLM_PROVIDER.
    """
    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=temperature,
        )

    elif settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{settings.LLM_PROVIDER}'. Use 'groq' or 'gemini'.")
