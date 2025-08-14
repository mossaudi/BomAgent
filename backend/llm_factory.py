# llm_factory_simplified.py - Gemini-Only Factory for Simplicity
"""
Simplified factory using only Gemini models to avoid complexity.
Removes Qwen2 and HuggingFace dependencies.
"""
from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI

from config import AppConfig


class LLMFactory:
    """Simplified factory using only Google Gemini models."""

    def __init__(self, config: AppConfig):
        self.config = config

    def create_vision_llm(self) -> BaseLanguageModel:
        """
        Creates the LLM for vision tasks (schematic analysis).
        Uses Gemini 1.5 Flash with vision capabilities.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.config.google_api_key,
            temperature=0.1,
            max_output_tokens=30000,
            convert_system_message_to_human=True
        )

    def create_agent_llm(self) -> BaseLanguageModel:
        """
        Creates the LLM for React agent reasoning.
        Uses Gemini 1.5 Flash optimized for structured output.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.config.google_api_key,
            temperature=0.0,  # Zero temperature for consistent parsing
            max_output_tokens=1024,  # Limited output for better parsing
            convert_system_message_to_human=True,  # Better React compatibility
            # Additional parameters for better React agent performance
            top_p=0.95,
            top_k=40
        )

    def create_lightweight_llm(self) -> BaseLanguageModel:
        """
        Creates a lightweight LLM for simple tasks.
        Same as agent but with even smaller output limit.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.config.google_api_key,
            temperature=0.0,
            max_output_tokens=512,  # Very limited for fast responses
            convert_system_message_to_human=True
        )


# Alternative LLM Options (if you want to replace Gemini)
class AlternativeLLMFactory:
    """Alternative LLM options if you want to move away from Gemini."""

    def __init__(self, config: AppConfig):
        self.config = config

    def create_openai_llm(self) -> BaseLanguageModel:
        """
        OpenAI GPT-4 option (requires OPENAI_API_KEY).
        Very reliable for React agents.
        """
        try:
            from langchain_openai import ChatOpenAI
            import os

            return ChatOpenAI(
                model="gpt-4o-mini",  # Cost-effective option
                temperature=0.0,
                max_tokens=1024,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")

    def create_anthropic_llm(self) -> BaseLanguageModel:
        """
        Anthropic Claude option (requires ANTHROPIC_API_KEY).
        Excellent for structured reasoning.
        """
        try:
            from langchain_anthropic import ChatAnthropic
            import os

            return ChatAnthropic(
                model="claude-3-haiku-20240307",  # Fast and cost-effective
                temperature=0.0,
                max_tokens=1024,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")

    def create_ollama_llm(self) -> BaseLanguageModel:
        """
        Local Ollama option (requires Ollama installed locally).
        Free but requires local setup.
        """
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model="llama3.1:8b",  # Good balance of size and performance
                temperature=0.0,
                num_predict=1024
            )
        except ImportError:
            raise ImportError("Install langchain-ollama: pip install langchain-ollama")


# Factory selector based on preference
def create_llm_factory(config: AppConfig, llm_provider: str = "gemini") -> LLMFactory:
    """
    Factory selector for different LLM providers.

    Args:
        config: Application configuration
        llm_provider: Choose from "gemini", "openai", "anthropic", "ollama"

    Returns:
        Appropriate LLM factory
    """
    if llm_provider.lower() == "gemini":
        return LLMFactory(config)
    else:
        return AlternativeLLMFactory(config)