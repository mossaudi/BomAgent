# container_simplified.py - Simplified Container
"""
Updated container using the simplified LLM factory.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from langchain_core.language_models import BaseLanguageModel

from src.clients.silicon_expert_client import SiliconExpertClient
from src.services.bom_service import BOMService
from src.services.component_service import ComponentService
from src.services.memory_service import MemoryService
from src.services.schematic_service import SchematicService
from llm_factory import LLMFactory, AlternativeLLMFactory, create_llm_factory

from config import AppConfig


@dataclass
class ServiceRegistry:
    """Registry of all application services."""
    memory: MemoryService
    schematic: SchematicService
    component: ComponentService
    bom: BOMService
    silicon_expert_client: SiliconExpertClient
    agent_llm: BaseLanguageModel
    vision_llm: BaseLanguageModel


class Container:
    """Simplified container with better LLM management."""

    def __init__(self, config: AppConfig, session_id: Optional[str] = None):
        self.silicon_expert_client = None
        self.config = config
        self.session_id = session_id
        self._services: Optional[ServiceRegistry] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all services with simplified LLM creation."""
        if self._initialized:
            return

        try:
            # Create LLM factory based on configuration
            llm_factory = create_llm_factory(self.config, self.config.llm_provider)

            # Create LLMs based on provider
            if isinstance(llm_factory, LLMFactory):
                # Gemini provider
                agent_llm = llm_factory.create_agent_llm()
                vision_llm = llm_factory.create_vision_llm()
            else:
                # Alternative providers - use same LLM for both tasks for simplicity
                if self.config.llm_provider == "openai":
                    agent_llm = llm_factory.create_openai_llm()
                    vision_llm = llm_factory.create_openai_llm()  # GPT-4 has vision
                elif self.config.llm_provider == "anthropic":
                    agent_llm = llm_factory.create_anthropic_llm()
                    vision_llm = llm_factory.create_anthropic_llm()  # Claude has vision
                elif self.config.llm_provider == "ollama":
                    agent_llm = llm_factory.create_ollama_llm()
                    vision_llm = llm_factory.create_ollama_llm()  # Use same for both
                else:
                    raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

            # Initialize Silicon Expert client
            silicon_expert_client = await self._create_silicon_expert_client()

            # Initialize services
            memory_service = MemoryService(session_id=self.session_id)
            schematic_service = SchematicService(vision_llm)
            component_service = ComponentService(silicon_expert_client, memory_service)
            bom_service = BOMService(silicon_expert_client, memory_service)

            self._services = ServiceRegistry(
                memory=memory_service,
                schematic=schematic_service,
                component=component_service,
                bom=bom_service,
                silicon_expert_client=silicon_expert_client,
                agent_llm=agent_llm,
                vision_llm=vision_llm
            )

            self._initialized = True
            print(f"✅ Container initialized with {self.config.llm_provider} LLM provider")

        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Container initialization failed: {str(e)}")

    async def _create_silicon_expert_client(self) -> SiliconExpertClient:
        """Create and authenticate Silicon Expert client."""
        client = SiliconExpertClient(self.config.silicon_expert)
        await client.authenticate()
        return client

    @property
    def services(self) -> ServiceRegistry:
        """Get the service registry."""
        if not self._initialized or not self._services:
            raise RuntimeError("Container not initialized. Call await container.initialize() first.")
        return self._services

    def get_agent_llm(self) -> BaseLanguageModel:
        """Get the LLM instance for the agent."""
        return self.services.agent_llm

    @asynccontextmanager
    async def session_scope(self):
        """Context manager for session lifecycle."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._services:
            try:
                await self._services.memory.cleanup()
                await self._services.silicon_expert_client.close()
            except Exception as e:
                print(f"Cleanup warning: {e}")
        self._initialized = False