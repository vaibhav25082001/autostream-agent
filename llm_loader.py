"""
llm_loader.py
=============
Loads the correct LLM based on the LLM_PROVIDER env variable.
Supported values: "claude" | "openai" | "gemini"
"""

import os
from langchain_core.language_models.chat_models import BaseChatModel


def load_llm() -> BaseChatModel:
    """
    Reads LLM_PROVIDER from .env and returns the matching LangChain LLM.

    Supported providers:
      claude  → Claude 3 Haiku   (Anthropic)
      openai  → GPT-4o-mini      (OpenAI)
      gemini  → Gemini 1.5 Flash (Google)
    """
    provider = os.getenv("LLM_PROVIDER", "claude").strip().lower()

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set in .env")
        print(f"[LLM] Using Claude 3 Haiku (Anthropic)")
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
            anthropic_api_key=api_key
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env")
        print(f"[LLM] Using GPT-4o-mini (OpenAI)")
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in .env")
        print(f"[LLM] Using Gemini 1.5 Flash (Google)")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=api_key
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Valid options: claude | openai | gemini"
        )
