# llm_factory_simplified.py - Gemini-Only Factory for Simplicity
"""
Simplified factory using only Gemini models to avoid complexity.
Removes Qwen2 and HuggingFace dependencies.
"""
from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI

from config import AppConfig


class LLMFactory:
    """Fixed factory with proper Gemini vision configuration."""

    def __init__(self, config: AppConfig):
        self.config = config

    def create_vision_llm(self) -> BaseLanguageModel:
        """
        Creates the LLM for vision tasks (schematic analysis).
        Uses Gemini Flash with proper vision configuration.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Use the experimental version for better vision
            google_api_key=self.config.google_api_key,
            temperature=0.1,  # Low temperature for consistent extraction
            max_output_tokens=32000,  # Large token limit for detailed analysis
            convert_system_message_to_human=False,  # Keep system messages as system
            # Vision-specific parameters
            top_p=0.95,
            top_k=40
        )

    def create_agent_llm(self) -> BaseLanguageModel:
        """
        Creates the LLM for ReAct agent reasoning.
        Uses Gemini Flash optimized for tool calling and structured output.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.config.google_api_key,
            temperature=0.0,  # Zero temperature for consistent tool calling
            max_output_tokens=2048,  # Reasonable limit for tool responses
            convert_system_message_to_human=False,  # Keep system context
            # Tool calling optimization
            top_p=0.95,
            top_k=20  # Lower K for more focused responses
        )

    def create_lightweight_llm(self) -> BaseLanguageModel:
        """
        Creates a lightweight LLM for simple tasks and quick responses.
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Use standard version for lightweight tasks
            google_api_key=self.config.google_api_key,
            temperature=0.0,
            max_output_tokens=1024,  # Very limited for fast responses
            convert_system_message_to_human=False,
            top_p=0.9,
            top_k=10
        )


# Alternative LLM Options for different providers
class AlternativeLLMFactory:
    """Alternative LLM options for non-Gemini providers."""

    def __init__(self, config: AppConfig):
        self.config = config

    def create_openai_llm(self) -> BaseLanguageModel:
        """
        OpenAI GPT-4 Vision option (requires OPENAI_API_KEY).
        Excellent for both vision and reasoning tasks.
        """
        try:
            from langchain_openai import ChatOpenAI
            import os

            return ChatOpenAI(
                model="gpt-4o",  # GPT-4 with vision capabilities
                temperature=0.0,
                max_tokens=4096,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                # Vision and tool calling optimized
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        except ImportError:
            raise ImportError("Install langchain-openai: pip install langchain-openai")

    def create_anthropic_llm(self) -> BaseLanguageModel:
        """
        Anthropic Claude option (requires ANTHROPIC_API_KEY).
        Excellent for structured reasoning and tool calling.
        """
        try:
            from langchain_anthropic import ChatAnthropic
            import os

            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022",  # Latest Claude with vision
                temperature=0.0,
                max_tokens=4096,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                # Optimized for tool calling
                top_p=0.95,
                top_k=20
            )
        except ImportError:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")

    def create_ollama_llm(self) -> BaseLanguageModel:
        """
        Local Ollama option with vision model.
        Requires Ollama with llama3.2-vision or similar model.
        """
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model="llama3.2-vision:11b",  # Vision-capable model
                temperature=0.0,
                num_predict=2048,
                # Local model optimization
                repeat_penalty=1.1,
                top_k=20,
                top_p=0.9
            )
        except ImportError:
            raise ImportError("Install langchain-ollama: pip install langchain-ollama")


def create_llm_factory(config: AppConfig, llm_provider: str = "gemini") -> LLMFactory:
    """
    Factory selector for different LLM providers with vision support.

    Args:
        config: Application configuration
        llm_provider: Choose from "gemini", "openai", "anthropic", "ollama"

    Returns:
        Appropriate LLM factory with vision capabilities
    """
    if llm_provider.lower() == "gemini":
        return LLMFactory(config)
    else:
        return AlternativeLLMFactory(config)