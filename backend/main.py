# clean_bom_agent.py - Professional ReAct Agent Implementation
"""
Clean, professional ReAct agent with proper separation of concerns:
1. Agent returns structured data, not UI formatting
2. Tools are simple and focused
3. Dynamic tool registration system
4. Proper async/await handling
5. Clean error handling and validation
"""

import asyncio
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Literal, Union
from datetime import datetime
from urllib.parse import urlparse
from dotmap import DotMap

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from pydantic import BaseModel


# =============================================================================
# Core Data Models
# =============================================================================

@dataclass
class ComponentData:
    """Clean component data structure"""
    id: str
    name: str
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    value: Optional[str] = None
    designator: Optional[str] = None
    confidence: float = 0.0
    enhanced: bool = False
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Schematic analysis result"""
    success: bool
    components: List[ComponentData]
    total_found: int
    enhanced_count: int
    analysis_url: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "components": [comp.to_dict() for comp in self.components],
            "total_found": self.total_found,
            "enhanced_count": self.enhanced_count,
            "enhancement_rate": (self.enhanced_count / self.total_found * 100) if self.total_found > 0 else 0,
            "analysis_url": self.analysis_url,
            "error": self.error
        }


@dataclass
class AgentResponse:
    """Standardized agent response for UI consumption"""
    success: bool
    action: str
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Agent State Management
# =============================================================================

class AgentState(BaseModel):
    """LangGraph state with proper typing"""
    messages: List[Any] = []
    session_id: str
    stored_components: List[ComponentData] = []
    last_analysis_url: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = []


# =============================================================================
# Clean Tool Implementations
# =============================================================================

class BaseTool:
    """Base class for all tools with consistent interface"""

    def __init__(self, agent: 'BOMAgent'):
        self.agent = agent

    def validate_input(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate tool input. Override in subclasses."""
        return True, None


class SchematicAnalysisTool(BaseTool):
    """Clean schematic analysis tool"""

    @tool
    async def analyze_schematic(self, image_url: str) -> Dict[str, Any]:
        """
        Analyze electronic schematic from image URL.
        Returns structured component data for UI rendering.

        Args:
            image_url: Public HTTP/HTTPS URL to schematic image

        Returns:
            Analysis result with components and metadata
        """
        # Validate URL
        is_valid, error = self._validate_url(image_url)
        if not is_valid:
            return AgentResponse(
                success=False,
                action="analyze_schematic",
                error=error
            ).to_dict()

        try:
            # Perform analysis
            result = await self._analyze_schematic(image_url)

            # Store components in agent state
            await self.agent.store_components(result.components)

            return AgentResponse(
                success=True,
                action="analyze_schematic",
                data=result.to_dict(),
                message=f"Successfully analyzed schematic. Found {result.total_found} components, enhanced {result.enhanced_count}"
            ).to_dict()

        except Exception as e:
            return AgentResponse(
                success=False,
                action="analyze_schematic",
                error=str(e)
            ).to_dict()

    def _validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate image URL format"""
        try:
            result = urlparse(url)
            if not all([result.scheme in ['http', 'https'], result.netloc, result.path]):
                return False, "Invalid URL format. Must be http:// or https://"
            return True, None
        except:
            return False, "Invalid URL format"

    async def _analyze_schematic(self, image_url: str) -> AnalysisResult:
        """Core analysis logic"""
        # Get raw analysis from LLM
        raw_result = await self.agent.schematic_service.analyze(image_url)

        if not raw_result.get('components'):
            return AnalysisResult(
                success=False,
                components=[],
                total_found=0,
                enhanced_count=0,
                analysis_url=image_url,
                error="No components found in schematic"
            )

        # Convert to ComponentData objects
        raw_components = []
        for comp_data in raw_result['components']:
            component = ComponentData(
                id=str(uuid.uuid4()),
                name=comp_data.get('name', 'Unknown'),
                part_number=comp_data.get('part_number'),
                manufacturer=comp_data.get('manufacturer'),
                description=comp_data.get('description', ''),
                value=comp_data.get('value'),
                designator=comp_data.get('designator'),
                confidence=float(comp_data.get('confidence', 0.5)),
                category=comp_data.get('category', 'other')
            )
            raw_components.append(component)

        # Auto-enhance with Silicon Expert
        enhanced_components = await self._enhance_components(raw_components)

        enhanced_count = sum(1 for comp in enhanced_components if comp.enhanced)

        return AnalysisResult(
            success=True,
            components=enhanced_components,
            total_found=len(enhanced_components),
            enhanced_count=enhanced_count,
            analysis_url=image_url
        )

    async def _enhance_components(self, components: List[ComponentData]) -> List[ComponentData]:
        """Enhance components with Silicon Expert data"""
        enhanced = []

        for component in components:
            try:
                # Search Silicon Expert
                search_data = {
                    "name": component.name,
                    "part_number": component.part_number,
                    "manufacturer": component.manufacturer,
                    "description": component.description
                }

                search_result = await self.agent.silicon_expert_client.search_component(search_data)

                if search_result.success and search_result.part_number:
                    # Create enhanced component
                    enhanced_comp = ComponentData(
                        id=component.id,
                        name=component.name,
                        part_number=search_result.part_number,
                        manufacturer=search_result.manufacturer or component.manufacturer,
                        description=search_result.description or component.description,
                        value=component.value,
                        designator=component.designator,
                        confidence=max(component.confidence, search_result.confidence),
                        enhanced=True,
                        category=component.category
                    )
                    enhanced.append(enhanced_comp)
                else:
                    # Keep original data
                    component.enhanced = False
                    enhanced.append(component)

            except Exception as e:
                # Fallback to original on error
                component.enhanced = False
                enhanced.append(component)

        return enhanced


class ComponentManagementTool(BaseTool):
    """Component management operations"""

    @tool
    async def get_stored_components(self) -> Dict[str, Any]:
        """
        Get all stored components from analysis.
        Returns structured data for UI table rendering.
        """
        try:
            components = await self.agent.get_stored_components()

            if not components:
                return AgentResponse(
                    success=True,
                    action="get_components",
                    data={"components": [], "total_count": 0},
                    message="No components available. Please analyze a schematic first."
                ).to_dict()

            enhanced_count = sum(1 for comp in components if comp.enhanced)

            return AgentResponse(
                success=True,
                action="get_components",
                data={
                    "components": [comp.to_dict() for comp in components],
                    "total_count": len(components),
                    "enhanced_count": enhanced_count,
                    "enhancement_rate": (enhanced_count / len(components) * 100) if components else 0
                },
                message=f"Retrieved {len(components)} components ({enhanced_count} enhanced)"
            ).to_dict()

        except Exception as e:
            return AgentResponse(
                success=False,
                action="get_components",
                error=str(e)
            ).to_dict()

    @tool
    async def search_component(self, query: str) -> Dict[str, Any]:
        """
        Search for component by name or part number.
        Returns component details from Silicon Expert.
        """
        if not query.strip():
            return AgentResponse(
                success=False,
                action="search_component",
                error="Search query cannot be empty"
            ).to_dict()

        try:
            search_data = {"name": query, "description": query}
            result = await self.agent.silicon_expert_client.search_component(search_data)

            if result.success and result.part_number:
                component_data = {
                    "part_number": result.part_number,
                    "manufacturer": result.manufacturer,
                    "description": result.description,
                    "lifecycle": result.lifecycle,
                    "confidence": result.confidence
                }

                return AgentResponse(
                    success=True,
                    action="search_component",
                    data={"component": component_data, "query": query},
                    message=f"Found component: {result.part_number}"
                ).to_dict()
            else:
                return AgentResponse(
                    success=True,
                    action="search_component",
                    data={"component": None, "query": query},
                    message=f"No component found for '{query}'"
                ).to_dict()

        except Exception as e:
            return AgentResponse(
                success=False,
                action="search_component",
                error=str(e)
            ).to_dict()


class BOMManagementTool(BaseTool):
    """BOM management operations"""

    @tool
    async def create_bom(self, name: str, project: str = "", description: str = "") -> Dict[str, Any]:
        """
        Create a new Bill of Materials.
        Returns BOM creation status and details.
        """
        if not name.strip():
            return AgentResponse(
                success=False,
                action="create_bom",
                error="BOM name is required"
            ).to_dict()

        try:
            result = await self.agent.bom_service.create_bom(
                name=name.strip(),
                project=project.strip(),
                description=description.strip()
            )

            if result.get('success'):
                components = await self.agent.get_stored_components()

                return AgentResponse(
                    success=True,
                    action="create_bom",
                    data={
                        "bom_name": name,
                        "project": project,
                        "description": description,
                        "available_components": len(components),
                        "api_response": result
                    },
                    message=f"BOM '{name}' created successfully"
                ).to_dict()
            else:
                return AgentResponse(
                    success=False,
                    action="create_bom",
                    error=result.get('error', 'Unknown error')
                ).to_dict()

        except Exception as e:
            return AgentResponse(
                success=False,
                action="create_bom",
                error=str(e)
            ).to_dict()

    @tool
    async def add_components_to_bom(self, bom_name: str) -> Dict[str, Any]:
        """
        Add all stored components to existing BOM.
        Returns operation status and component details.
        """
        if not bom_name.strip():
            return AgentResponse(
                success=False,
                action="add_components_to_bom",
                error="BOM name is required"
            ).to_dict()

        try:
            components = await self.agent.get_stored_components()

            if not components:
                return AgentResponse(
                    success=False,
                    action="add_components_to_bom",
                    error="No components available. Please analyze a schematic first."
                ).to_dict()

            # Convert to BOM format
            parts_data = []
            for comp in components:
                part = {
                    "part_number": comp.part_number or comp.designator or "Unknown",
                    "manufacturer": comp.manufacturer or "Unknown",
                    "description": f"{comp.name} - {comp.description or comp.value or ''}".strip(" - "),
                    "quantity": "1",
                    "designator": comp.designator or ""
                }
                parts_data.append(part)

            result = await self.agent.bom_service.add_parts(bom_name.strip(), "", parts_data)

            if result.get('success'):
                enhanced_count = sum(1 for comp in components if comp.enhanced)

                return AgentResponse(
                    success=True,
                    action="add_components_to_bom",
                    data={
                        "bom_name": bom_name,
                        "total_components": len(components),
                        "enhanced_components": enhanced_count,
                        "original_components": len(components) - enhanced_count,
                        "api_response": result
                    },
                    message=f"Added {len(components)} components to '{bom_name}'"
                ).to_dict()
            else:
                return AgentResponse(
                    success=False,
                    action="add_components_to_bom",
                    error=result.get('error', 'Unknown error')
                ).to_dict()

        except Exception as e:
            return AgentResponse(
                success=False,
                action="add_components_to_bom",
                error=str(e)
            ).to_dict()

    @tool
    async def list_boms(self, project_filter: str = "") -> Dict[str, Any]:
        """
        List all existing BOMs with optional project filter.
        Returns BOM list for UI rendering.
        """
        try:
            result = await self.agent.bom_service.get_boms(project_filter)

            if result.get('success'):
                boms = result.get('boms', [])
                components = await self.agent.get_stored_components()

                return AgentResponse(
                    success=True,
                    action="list_boms",
                    data={
                        "boms": boms,
                        "total_boms": len(boms),
                        "available_components": len(components),
                        "project_filter": project_filter
                    },
                    message=f"Retrieved {len(boms)} BOMs"
                ).to_dict()
            else:
                return AgentResponse(
                    success=False,
                    action="list_boms",
                    error=result.get('error', 'Unknown error')
                ).to_dict()

        except Exception as e:
            return AgentResponse(
                success=False,
                action="list_boms",
                error=str(e)
            ).to_dict()


# =============================================================================
# Dynamic Tool Registry
# =============================================================================

class ToolRegistry:
    """Dynamic tool registration system"""

    def __init__(self, agent: 'BOMAgent'):
        self.agent = agent
        self._tools = {}
        self._tool_instances = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""
        self.register_tool_class("schematic", SchematicAnalysisTool)
        self.register_tool_class("component", ComponentManagementTool)
        self.register_tool_class("bom", BOMManagementTool)

    def register_tool_class(self, category: str, tool_class: type):
        """Register a tool class for dynamic instantiation"""
        self._tools[category] = tool_class

    def get_all_tools(self) -> List:
        """Get all registered tool functions"""
        tools = []

        for category, tool_class in self._tools.items():
            if category not in self._tool_instances:
                self._tool_instances[category] = tool_class(self.agent)

            instance = self._tool_instances[category]

            # Get all methods decorated with @tool
            for attr_name in dir(instance):
                attr = getattr(instance, attr_name)
                if hasattr(attr, 'name') and hasattr(attr, 'description'):
                    tools.append(attr)

        return tools

    def add_custom_tool(self, tool_func):
        """Add a custom tool function at runtime"""
        # This allows for dynamic tool addition
        return tool_func


# =============================================================================
# Clean ReAct Agent
# =============================================================================

class BOMAgent:
    """Professional ReAct agent for BOM management"""

    def __init__(self, config, session_id: str = None):
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation_history: List[Dict[str, Any]] = []

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=config.google_api_key,
            temperature=0.1,
            max_output_tokens=4000
        )

        # Component storage
        self._stored_components: List[ComponentData] = []

        # Services (injected or created)
        self.schematic_service = None
        self.component_service = None
        self.bom_service = None
        self.silicon_expert_client = None

        # Tool registry
        self.tool_registry = ToolRegistry(self)

        # LangGraph setup
        self.graph = None
        self._build_graph()

    async def initialize(self) -> None:
        """Initialize agent services"""
        # Initialize services here
        # This method now exists for server.py compatibility
        pass

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.silicon_expert_client:
            await self.silicon_expert_client.close()

    def _build_graph(self):
        """Build the ReAct workflow graph"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        # Compile
        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)

    async def _agent_node(self, state: AgentState) -> AgentState:
        """Main reasoning node"""
        system_msg = SystemMessage(content=f"""You are a professional electronic design BOM assistant.

**Your Capabilities:**
- Analyze electronic schematics and extract components
- Search for component specifications using Silicon Expert
- Create and manage Bills of Materials (BOMs)

**Available Tools:**
{self._get_tool_descriptions()}

**Current Context:**
- Session: {state.session_id}
- Stored Components: {len(state.stored_components)} available

**Instructions:**
1. For schematic analysis: extract URL and use analyze_schematic tool
2. Always return structured data, not formatted text
3. Chain tools when appropriate
4. Be helpful and provide clear responses
5. Use tool calls in proper JSON format when needed

Focus on understanding intent and providing structured responses.""")

        # Prepare messages
        messages = state.messages.copy()
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, system_msg)

        # Get LLM response
        agent = self.llm.bind_tools(self.tool_registry.get_all_tools())
        response = await agent.ainvoke(messages)

        # Update state
        new_state = state.copy()
        new_state.messages = messages + [response]

        return new_state

    async def _tool_node(self, state: AgentState) -> AgentState:
        """Execute tool calls"""
        last_message = state.messages[-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            results = []
            for tool_call in last_message.tool_calls:
                result = await self._execute_tool_call(tool_call)
                results.append(result)

            # Update state with results
            new_state = state.copy()
            for result in results:
                new_state.messages.append(AIMessage(content=str(result)))

            return new_state

        return state

    async def _execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute a specific tool call"""
        tool_name = tool_call['name']
        tool_args = tool_call.get('args', {})

        # Find and execute tool
        for tool in self.tool_registry.get_all_tools():
            if tool.name == tool_name:
                try:
                    result = await tool.func(**tool_args)
                    return result
                except Exception as e:
                    return {
                        "success": False,
                        "action": tool_name,
                        "error": str(e)
                    }

        return {
            "success": False,
            "action": tool_name,
            "error": f"Tool '{tool_name}' not found"
        }

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if workflow should continue"""
        last_message = state.messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"

    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions"""
        tools = self.tool_registry.get_all_tools()
        descriptions = [f"- {tool.name}: {tool.description}" for tool in tools]
        return "\n".join(descriptions)

    # =============================================================================
    # Public Interface Methods (for server.py compatibility)
    # =============================================================================

    async def process_request(self, message: str) -> AgentResponse:
        """Process user request and return structured response"""
        try:
            # Create initial state
            initial_state = AgentState(
                messages=[HumanMessage(content=message)],
                session_id=self.session_id,
                stored_components=self._stored_components
            )

            # Run workflow
            config = {"configurable": {"thread_id": self.session_id}}
            final_state = await self.graph.ainvoke(initial_state, config)

            final_state = DotMap(final_state)

            # Update stored components
            self._stored_components = final_state.stored_components

            # Add to conversation history
            self.conversation_history.append({
                "type": "human",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })

            # Extract final response
            if final_state.messages:
                last_message = final_state.messages[-1]
                if isinstance(last_message, AIMessage):
                    response_content = last_message.content

                    # Add to conversation history
                    self.conversation_history.append({
                        "type": "assistant",
                        "content": response_content,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Try to parse as structured response
                    if isinstance(response_content, dict):
                        return AgentResponse(**response_content)
                    else:
                        return AgentResponse(
                            success=True,
                            action="general_response",
                            message=str(response_content)
                        )

            return AgentResponse(
                success=True,
                action="general_response",
                message="I'm here to help with electronic design and BOM management."
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                action="error",
                error=str(e)
            )

    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history.copy()

    async def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()

    async def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            "session_id": self.session_id,
            "history_count": len(self.conversation_history),
            "components_count": len(self._stored_components),
            "initialized": True,
            "last_activity": datetime.now().isoformat() if self.conversation_history else None
        }

    async def store_components(self, components: List[ComponentData]) -> None:
        """Store components in agent memory"""
        self._stored_components = components

    async def get_stored_components(self) -> List[ComponentData]:
        """Get stored components"""
        return self._stored_components.copy()


# =============================================================================
# Usage Example
# =============================================================================

async def main():
    """Example usage"""
    from config import AppConfig

    config = AppConfig.from_env()
    agent = BOMAgent(config)

    # Initialize agent
    await agent.initialize()

    # Test conversation
    response = await agent.process_request("Analyze schematic at https://example.com/schematic.png")
    print("Response:", response.to_dict())

    # Get components
    response = await agent.process_request("Show me the components you found")
    print("Components:", response.to_dict())

    # Cleanup
    await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())