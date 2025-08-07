# container.py - ENHANCED WITH MEMORY
"""Enhanced container with memory management integration."""

from langchain_google_genai import ChatGoogleGenerativeAI

from services.formatter import ComponentTableFormatter
from services.memory import get_memory_manager, configure_memory_manager
from config import AppConfig
from clients.silicon_expert import SiliconExpertClient
from services.analysis import ComponentAnalysisService
from services.workflow import BOMWorkflowService
from services.bom_management import BOMManagementService
from services.parsing import ParsingService


class Container:
    """Enhanced container for managing and injecting service dependencies with memory support."""

    def __init__(self, config: AppConfig, llm: ChatGoogleGenerativeAI, session_id: str = None):
        self.config = config
        self.llm = llm

        # Initialize memory management
        self._setup_memory(session_id)

        # --- Clients ---
        self.silicon_expert_client = SiliconExpertClient(config.silicon_expert)

        # --- Services ---
        self.parsing_service = ParsingService()
        self.analysis_service = ComponentAnalysisService(llm, self.silicon_expert_client)
        self.bom_service = BOMManagementService(self.silicon_expert_client)
        self.workflow_service = BOMWorkflowService(
            analysis_service=self.analysis_service,
            parsing_service=self.parsing_service,
            silicon_expert_client=self.silicon_expert_client
        )
        self.formatter = ComponentTableFormatter()

        # Memory services
        self.memory_manager = get_memory_manager()

    def _setup_memory(self, session_id: str = None):
        """Setup memory management for this container instance."""
        # Configure memory with longer TTL for component data
        configure_memory_manager(default_ttl=7200)  # 2 hours

        memory_manager = get_memory_manager()

        if session_id:
            memory_manager.set_session(session_id)
        elif not memory_manager.get_session():
            # Create new session if none exists
            new_session_id = memory_manager.create_session()
            print(f"📝 Created new container session: {new_session_id}")

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.memory_manager.get_session()

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for this container."""
        self.memory_manager.set_session(session_id)

    def get_memory_summary(self) -> dict:
        """Get a summary of the current session's memory state."""
        return self.memory_manager.get_session_summary()