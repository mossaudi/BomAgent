# main.py - SESSION-AWARE VERSION
"""Enhanced LangGraph agent with session-based memory for multi-user support."""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src" / "agent"
sys.path.insert(0, str(src_path))

import getpass
import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.agent.config import AppConfig
from src.agent.container import Container
from src.agent.exceptions import ConfigurationError
from src.agent.models import SearchResult, BOMTreeResult
from src.agent.services.formatter import ComponentTableFormatter
from src.agent.services.memory import get_memory_manager, get_component_memory
from src.agent.tools import get_tools, get_memory_status


class IntelligentBOMAgent:
    """Enhanced LangGraph agent with session-based memory for multi-user support."""

    def __init__(self, config: AppConfig, session_id: Optional[str] = None):
        self.config = config
        self.session_id = session_id
        self._validate_config()
        self._initialize_components()
        self._create_react_agent()
        print(f"✅ Intelligent BOM Agent is ready (Session: {self.get_session_id()})")

    def _validate_config(self):
        config_errors = self.config.validate()
        if config_errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"- {error}" for error in config_errors)
            raise ConfigurationError(error_msg)

    def _initialize_components(self):
        """Initialize LLM, container, tools, and formatter with session support."""
        google_api_key = self.config.google_api_key or getpass.getpass("Enter API key for Google Gemini: ")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=30000,
        )
        print("✅ Gemini model configured.")

        # Initialize container with session support
        self.container = Container(self.config, self.llm, self.session_id)
        self.tools = get_tools(self.container)
        self.formatter = ComponentTableFormatter()

        # Get session info
        self.memory_manager = get_memory_manager()
        self.component_memory = get_component_memory()

        print(f"✅ Initialized {len(self.tools)} tools with session support.")

        self._setup_direct_handlers()
        print(f"✅ Registered {len(self.direct_handlers.handlers)} direct handlers.")

    def _create_react_agent(self):
        """Create ReAct agent with session-aware system prompt."""

        system_prompt = f"""You are an expert Bill of Materials (BOM) management assistant with session-based memory support. Your goal is to help users analyze electronic schematics, find component data, and manage BOMs efficiently.

CURRENT SESSION: {self.get_session_id()}

## KEY CAPABILITIES:
1. **Schematic Analysis**: Use `analyze_schematic` to extract and enhance component data from schematics
2. **Session Memory**: Components are stored per session - each user has isolated memory
3. **BOM Management**: Create BOMs and add components without requiring JSON from users
4. **Memory Management**: Use `get_memory_status` to check what's available in current session

## MEMORY SYSTEM:
- Each session has isolated memory (multi-user safe)
- Components from `analyze_schematic` are automatically stored
- Use `get_last_components` to retrieve stored components
- Use `add_parts_to_bom` without JSON to use stored components
- Use `get_memory_status` to see what's in memory
- Use `clear_memory(confirm=True)` to reset session

## TOOL BEHAVIOR:
- `analyze_schematic`: Returns formatted table AND stores in session memory
- `get_last_components`: Shows components from current session
- `add_parts_to_bom`: Auto-uses session components if no JSON provided
- `get_memory_status`: Shows current session's memory contents
- All tools support optional session_id parameter for multi-user scenarios

## WORKFLOW EXAMPLES:
1. User: "Analyze schematic at [URL]" 
   → `analyze_schematic` → Display results + store in session

2. User: "Create BOM from those components" 
   → `add_parts_to_bom` (auto-uses session components)

3. User: "What do I have in memory?" 
   → `get_memory_status` → Show session summary

## RESPONSE STRATEGY:
- Pass tool results directly (they're pre-formatted)
- Never ask users for JSON - tools handle data automatically  
- Mention session benefits for multi-user scenarios
- Suggest memory commands when relevant

## SESSION ISOLATION:
- Each session ID creates isolated component storage
- Multiple users can work simultaneously without interference
- Sessions persist for 2 hours by default
- Tools automatically manage session context

Remember: This agent supports multiple concurrent users through session isolation. Each user's components and analysis results are kept separate and secure."""

        # Use session-aware checkpointer
        checkpointer = MemorySaver()
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=checkpointer
        )

        self.system_prompt = system_prompt
        # Use session ID in agent config for LangGraph state isolation
        self.config_dict = {"configurable": {"thread_id": self.get_session_id()}}
        print("✅ Session-aware ReAct agent created.")

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.container.get_session_id()

    def set_session_id(self, session_id: str) -> None:
        """Switch to a different session."""
        self.container.set_session_id(session_id)
        self.session_id = session_id
        # Update agent config
        self.config_dict = {"configurable": {"thread_id": session_id}}
        print(f"🔄 Switched to session: {session_id}")

    def create_new_session(self) -> str:
        """Create a new session and switch to it."""
        new_session_id = self.memory_manager.create_session()
        self.set_session_id(new_session_id)
        return new_session_id

    def get_memory_summary(self) -> dict:
        """Get current session's memory summary."""
        return self.container.get_memory_summary()

    def _format_final_response(self, content: Any) -> str:
        """Enhanced response formatting that handles already-formatted content."""
        # If content is already a formatted string (from tools), return as-is
        if isinstance(content, str):
            # Check if it looks like it's already formatted
            if any(marker in content for marker in ["│", "├", "└", "=" * 10, "COMPONENT ANALYSIS", "SESSION MEMORY"]):
                return content

            # Try to parse as JSON for fallback formatting
            try:
                data = json.loads(content)
                if isinstance(data, dict) and 'components' in data and 'success_rate' in data:
                    search_result = SearchResult(**data)
                    return self.formatter.format_search_result(search_result)
                return json.dumps(data, indent=2)
            except (json.JSONDecodeError, TypeError):
                return content

        elif isinstance(content, dict):
            # Handle dictionary responses (mostly from BOM operations)
            if 'components' in content and 'success_rate' in content:
                search_result = SearchResult(**content)
                return self.formatter.format_search_result(search_result)
            return json.dumps(content, indent=2)

        return str(content)

    def process_request(self, user_input: str, user_id: str = None):
        """Process user request with session isolation support."""
        # If user_id provided, use it as session prefix for better organization
        if user_id and not self.get_session_id().startswith(user_id):
            session_id = f"{user_id}_{self.get_session_id()}"
            self.set_session_id(session_id)

        print(f"\n{'=' * 80}")
        print(f"🤖 PROCESSING REQUEST (Session: {self.get_session_id()})")
        print(f"👤 Request: {user_input}")
        print(f"{'=' * 80}")

        try:
            # Check for direct handlers first
            handler_func, description = self.direct_handlers.find_handler(user_input)

            if handler_func:
                print(f"🚀 Using direct handler: {description}")
                result = handler_func(user_input)
                print(f"\n🤖 Assistant:\n{result}")
                print(f"\n{'=' * 80}\n✨ REQUEST COMPLETED (Direct)\n{'=' * 80}")
                return

            # Use ReAct agent for complex reasoning tasks
            print("🧠 Using ReAct agent for complex processing...")
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_input)]
            result = self.agent.invoke({"messages": messages}, self.config_dict)

            if result and "messages" in result:
                final_message = result["messages"][-1]
                formatted_output = self._format_final_response(final_message.content)
                print(f"\n🤖 Assistant:\n{formatted_output}")
            else:
                print("\n🤖 Assistant: I apologize, but I didn't receive a proper response. Please try again.")

            print(f"\n{'=' * 80}\n✨ REQUEST COMPLETED (Agent)\n{'=' * 80}")

        except Exception as e:
            print(f"❌ Error processing request: {e}")
            if "quota" in str(e).lower() or "429" in str(e):
                print("💡 Tip: You've hit the API quota limit. Try again later or use direct commands.")

    def run_interactive(self, user_id: str = None):
        """Run the agent in an interactive command-line mode with session support."""
        session_id = self.get_session_id()
        if user_id:
            session_id = f"{user_id}_{session_id}"
            self.set_session_id(session_id)

        print(f"\n🤖 Welcome to the Intelligent BOM Agent!")
        print(f"🔐 Your Session ID: {session_id}")
        print("   Your components and analysis results are private to your session.")
        print("\n   Example commands:")
        print("   • 'analyze schematic at [URL]'")
        print("   • 'show me my memory status'")
        print("   • 'show me the last components'")
        print("   • 'create BOM called MyProject'")
        print("   • 'add components to BOM MyProject'")
        print("   • 'clear my memory' (to reset session)")
        print("   • Type 'quit' or 'exit' to end.")
        print("-" * 80)

        while True:
            try:
                user_input = input(f"\n👤 User ({session_id[:8]}...): ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("🤖 Goodbye! Your session data will be preserved for 2 hours.")
                    break
                self.process_request(user_input, user_id)
            except KeyboardInterrupt:
                print("\n🤖 Goodbye! Your session data will be preserved for 2 hours.")
                break
            except Exception as e:
                print(f"❌ An unexpected error occurred: {e}")

    def _setup_direct_handlers(self):
        """Setup direct handlers for operations that should bypass LLM."""
        self.direct_handlers = DirectHandlerRegistry()

        # Register memory status as direct handler
        self.direct_handlers.register(
            keywords=['memory status', 'show memory', 'what do I have', 'my memory', 'session status'],
            handler_func=self._handle_memory_status,
            description="Memory status check (direct)"
        )

        # Register BOM listing as direct handler
        self.direct_handlers.register(
            keywords=['list bom', 'show bom', 'get bom', 'view bom', 'display bom', 'my bom'],
            handler_func=self._handle_bom_listing,
            description="BOM listing (direct)"
        )

    def _handle_memory_status(self, user_input: str) -> str:
        """Handle memory status requests directly."""
        try:
            return get_memory_status()
        except Exception as e:
            return f"❌ Error retrieving memory status: {str(e)}"

    def _handle_bom_listing(self, user_input: str) -> str:
        """Handle BOM listing requests directly."""
        try:
            bom_result = self.container.bom_service.get_boms()

            if not bom_result.get("success", False):
                return f"❌ Failed to retrieve BOMs: {bom_result.get('raw_api_response', {}).get('Status', {}).get('message', 'Unknown error')}"

            bom_tree = BOMTreeResult(**bom_result["bom_tree"])
            return self.container.formatter.format_bom_tree(bom_tree)

        except Exception as e:
            return f"❌ Error retrieving BOMs: {str(e)}"


class DirectHandlerRegistry:
    """Registry for operations that should bypass LLM processing."""

    def __init__(self):
        self.handlers = {}

    def register(self, keywords: list, handler_func, description: str = ""):
        """Register a direct handler for specific keywords."""
        for keyword in keywords:
            self.handlers[keyword.lower()] = {
                'handler': handler_func,
                'description': description
            }

    def find_handler(self, user_input: str):
        """Find matching handler for user input."""
        user_lower = user_input.lower()
        for keyword, handler_info in self.handlers.items():
            if keyword in user_lower:
                return handler_info['handler'], handler_info['description']
        return None, None


# Multi-user application wrapper
class MultiUserBOMService:
    """Service wrapper for handling multiple concurrent users."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.active_agents = {}  # user_id -> agent instance

    def get_or_create_agent(self, user_id: str) -> IntelligentBOMAgent:
        """Get existing agent for user or create new one."""
        if user_id not in self.active_agents:
            # Create agent with user-specific session
            agent = IntelligentBOMAgent(self.config)
            session_id = f"{user_id}_{agent.get_session_id()}"
            agent.set_session_id(session_id)
            self.active_agents[user_id] = agent
            print(f"✅ Created new agent for user: {user_id}")

        return self.active_agents[user_id]

    def process_user_request(self, user_id: str, request: str) -> str:
        """Process a request from a specific user."""
        agent = self.get_or_create_agent(user_id)

        # Capture output for returning as string
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            agent.process_request(request, user_id)
            return captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

    def get_user_memory_status(self, user_id: str) -> dict:
        """Get memory status for a specific user."""
        if user_id in self.active_agents:
            return self.active_agents[user_id].get_memory_summary()
        return {'error': 'No active session for user'}

    def cleanup_inactive_sessions(self):
        """Clean up inactive user sessions (call periodically)."""
        # Implementation would check session expiry and remove inactive agents
        pass


def create_agent_graph(config: dict):
    """Factory function for LangGraph to create the agent graph."""
    app_config = AppConfig.from_env()

    agent = IntelligentBOMAgent(app_config)
    return agent.agent


def main():
    """Main entry point with multi-user support option."""
    try:
        config = AppConfig.from_env()

        # Check if running in multi-user mode
        import os
        if os.getenv("MULTI_USER_MODE", "").lower() == "true":
            # Multi-user service mode
            service = MultiUserBOMService(config)
            print("🌐 Multi-user BOM service started")
            # In real implementation, this would be a web server or API
            return 0
        else:
            # Single-user interactive mode
            user_id = os.getenv("USER_ID", "default_user")
            agent = IntelligentBOMAgent(config)
            agent.run_interactive(user_id)

    except (ConfigurationError, Exception) as e:
        print(f"❌ Fatal Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())