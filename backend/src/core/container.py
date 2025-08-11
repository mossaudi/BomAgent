# container.py - ENHANCED WITH MEMORY
"""Modern dependency injection container with clean architecture."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.src.clients.silicon_expert_client import SiliconExpertClient
from backend.src.core.config import AppConfig
from backend.src.services.bom_service import BOMService
from backend.src.services.component_service import ComponentService
from backend.src.services.memory_service import MemoryService
from backend.src.services.schematic_service import SchematicService


@dataclass
class ServiceRegistry:
    """Registry of all application services."""
    memory: MemoryService
    schematic: SchematicService
    component: ComponentService
    bom: BOMService
    silicon_expert_client: SiliconExpertClient
    llm: ChatGoogleGenerativeAI


class Container:
    """Modern dependency injection container with async support."""

    def __init__(self, config: AppConfig, session_id: Optional[str] = None):
        self.config = config
        self.session_id = session_id
        self._services: Optional[ServiceRegistry] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all services asynchronously."""
        if self._initialized:
            return

        # Initialize core components
        llm = self._create_llm()
        silicon_expert_client = await self._create_silicon_expert_client()

        # Initialize services
        memory_service = MemoryService(session_id=self.session_id)
        schematic_service = SchematicService(llm)
        component_service = ComponentService(silicon_expert_client, memory_service)
        bom_service = BOMService(silicon_expert_client, memory_service)

        self._services = ServiceRegistry(
            memory=memory_service,
            schematic=schematic_service,
            component=component_service,
            bom=bom_service,
            silicon_expert_client=silicon_expert_client,
            llm=llm
        )

        self._initialized = True

    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create and configure the LLM."""
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=self.config.google_api_key,
            temperature=0.1,
            max_output_tokens=30000,
        )

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

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """Get the LLM instance."""
        return self.services.llm

    def get_memory_service(self) -> MemoryService:
        """Get the memory service."""
        return self.services.memory

    def get_schematic_service(self) -> SchematicService:
        """Get the schematic service."""
        return self.services.schematic

    def get_component_service(self) -> ComponentService:
        """Get the component service."""
        return self.services.component

    def get_bom_service(self) -> BOMService:
        """Get the BOM service."""
        return self.services.bom

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
            await self._services.memory.cleanup()
            await self._services.silicon_expert_client.close()