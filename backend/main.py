# optimized_bom_agent.py - Clean ReAct Agent Implementation
"""
Optimized ReAct agent addressing all identified issues:
1. Clean code architecture with proper separation of concerns
2. Single source of truth for models and services
3. Dynamic tool system with clear parameter definitions
4. Fixed tool execution issues
5. Simplified and maintainable codebase
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.core.bom_tools import ComponentStateManager
from src.core.container import Container
from src.core.models import ComponentData, AgentResponse


# =============================================================================
# Core Models - Single Source of Truth
# =============================================================================

@dataclass
class AgentResponse:
    """Single response model for all operations"""
    success: bool
    action: str
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Tool System - Fixed and Enhanced
# =============================================================================

class SchematicAnalysisInput(BaseModel):
    """Input schema for schematic analysis tool"""
    image_url: str = Field(description="Valid HTTP/HTTPS URL to schematic image")


class ComponentSearchInput(BaseModel):
    """Input schema for component search tool"""
    query: str = Field(description="Component name, part number, or description to search for")


class BOMCreateInput(BaseModel):
    """Input schema for BOM creation tool"""
    name: str = Field(description="BOM name (required)")
    project: str = Field(default="", description="Project name (optional)")
    description: str = Field(default="", description="BOM description (optional)")


class BOMAddInput(BaseModel):
    """Input schema for adding components to BOM"""
    bom_name: str = Field(description="Name of the BOM to add components to")


class AgentTool(ABC):
    """Base class for all agent tools"""

    def __init__(self, agent: 'BOMAgent'):
        self.agent = agent

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResponse:
        """Execute the tool with given parameters"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for registration"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM"""
        pass

    @property
    @abstractmethod
    def args_schema(self):
        """Tool argument schema"""
        pass


class SchematicAnalysisTool(AgentTool):
    """Tool for analyzing schematics with proper schema"""

    @property
    def name(self) -> str:
        return "analyze_schematic"

    @property
    def description(self) -> str:
        return """Analyze electronic schematic from image URL and extract all visible components.

        Input: image_url (string) - Valid HTTP/HTTPS URL to schematic image
        Output: List of components with enhanced details from Silicon Expert database

        Example: analyze_schematic(image_url="https://example.com/schematic.png")"""

    @property
    def args_schema(self):
        return SchematicAnalysisInput

    async def execute(self, image_url: str) -> AgentResponse:
        """Execute schematic analysis with proper parameter handling"""
        # Validate URL
        if not self._is_valid_url(image_url):
            return AgentResponse(
                success=False,
                action=self.name,
                error="Invalid URL format. Must be http:// or https://",
                message="Please provide a valid HTTP/HTTPS URL to your schematic image."
            )

        try:
            # Analyze schematic
            analysis_result = await self.agent.container.services.schematic.analyze_with_retry(image_url)

            if not analysis_result.get('components'):
                return AgentResponse(
                    success=False,
                    action=self.name,
                    error="No components found in schematic",
                    message="No components could be extracted from the schematic image."
                )

            # Convert to ComponentData objects
            raw_components = []
            for comp_data in analysis_result['components']:
                component = ComponentData(
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

            # Store raw components
            await self.agent.container.services.memory.store_components(raw_components)

            # Auto-enhance components
            enhanced_components = await self.agent.container.services.component.search_and_enhance(raw_components)

            # Store enhanced components
            self.agent.component_state.store_enhanced_components(enhanced_components)

            enhanced_count = sum(1 for comp in enhanced_components if comp.enhanced)

            return AgentResponse(
                success=True,
                action=self.name,
                message=f"Successfully analyzed schematic. Found {len(enhanced_components)} components, enhanced {enhanced_count} with Silicon Expert data.",
                data={
                    "components": [comp.to_dict() for comp in enhanced_components],
                    "total_found": len(enhanced_components),
                    "enhanced_count": enhanced_count,
                    "enhancement_rate": (enhanced_count / len(enhanced_components) * 100) if enhanced_components else 0
                }
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                action=self.name,
                error=str(e),
                message=f"Schematic analysis failed: {str(e)}"
            )

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc, result.path])
        except:
            return False


class ComponentSearchTool(AgentTool):
    """Tool for searching components"""

    @property
    def name(self) -> str:
        return "search_component"

    @property
    def description(self) -> str:
        return """Search for electronic component by name or part number using Silicon Expert database.

        Input: query (string) - Component name, part number, or description
        Output: Component details including part number, manufacturer, description, lifecycle

        Example: search_component(query="LM358")"""

    @property
    def args_schema(self):
        return ComponentSearchInput

    async def execute(self, query: str) -> AgentResponse:
        """Execute component search"""
        if not query or not query.strip():
            return AgentResponse(
                success=False,
                action=self.name,
                error="Search query cannot be empty",
                message="Please provide a component name or part number to search for."
            )

        try:
            search_data = {"name": query.strip(), "description": query.strip()}
            result_list = await self.agent.container.services.component.search_and_enhance([search_data])

            result = result_list[0] or {}
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
                    action=self.name,
                    message=f"Found component: {result.part_number} by {result.manufacturer}",
                    data={"component": component_data, "query": query}
                )
            else:
                return AgentResponse(
                    success=True,
                    action=self.name,
                    message=f"No matching component found for '{query}'",
                    data={"component": None, "query": query}
                )

        except Exception as e:
            return AgentResponse(
                success=False,
                action=self.name,
                error=str(e),
                message=f"Component search failed: {str(e)}"
            )


class BOMCreateTool(AgentTool):
    """Tool for creating BOMs"""

    @property
    def name(self) -> str:
        return "create_bom"

    @property
    def description(self) -> str:
        return """Create a new Bill of Materials (BOM) in Silicon Expert.

        Input: 
        - name (string, required) - BOM name
        - project (string, optional) - Project name  
        - description (string, optional) - BOM description

        Output: Confirmation of BOM creation with available components count

        Example: create_bom(name="Arduino Shield", project="IoT Device", description="Main PCB BOM")"""

    @property
    def args_schema(self):
        return BOMCreateInput

    async def execute(self, name: str, project: str = "", description: str = "") -> AgentResponse:
        """Execute BOM creation"""
        if not name or not name.strip():
            return AgentResponse(
                success=False,
                action=self.name,
                error="BOM name is required",
                message="Please provide a name for the BOM."
            )

        try:
            result = await self.agent.container.services.bom.create_bom(
                name=name.strip(),
                description=description.strip(),
                project=project.strip()
            )

            if result.get('success'):
                components_count = len(self.agent.component_state.get_components_for_bom())

                return AgentResponse(
                    success=True,
                    action=self.name,
                    message=f"BOM '{name}' created successfully. {components_count} components available to add.",
                    data={
                        "bom_name": name,
                        "project": project,
                        "description": description,
                        "available_components": components_count
                    }
                )
            else:
                return AgentResponse(
                    success=False,
                    action=self.name,
                    error=result.get('error', 'Unknown error'),
                    message=f"Failed to create BOM '{name}'"
                )

        except Exception as e:
            return AgentResponse(
                success=False,
                action=self.name,
                error=str(e),
                message=f"BOM creation failed: {str(e)}"
            )


class BOMAddComponentsTool(AgentTool):
    """Tool for adding components to BOM"""

    @property
    def name(self) -> str:
        return "add_components_to_bom"

    @property
    def description(self) -> str:
        return """Add all stored components to an existing BOM.

        Input: bom_name (string) - Name of the BOM to add components to
        Output: Confirmation of components added to BOM

        Example: add_components_to_bom(bom_name="Arduino Shield")

        Note: Components must be analyzed first using analyze_schematic"""

    @property
    def args_schema(self):
        return BOMAddInput

    async def execute(self, bom_name: str) -> AgentResponse:
        """Execute adding components to BOM"""
        if not bom_name or not bom_name.strip():
            return AgentResponse(
                success=False,
                action=self.name,
                error="BOM name is required",
                message="Please provide the name of the BOM to add components to."
            )

        components = self.agent.component_state.get_components_for_bom()

        if not components:
            return AgentResponse(
                success=False,
                action=self.name,
                error="No components available",
                message="No components found. Please analyze a schematic first using analyze_schematic."
            )

        try:
            # Convert to API format
            parts_data = []
            for comp in components:
                part = {
                    "part_number": comp.part_number or comp.designator or "Unknown",
                    "manufacturer": comp.manufacturer or "Unknown",
                    "description": f"{comp.name} - {comp.description or comp.value or ''}".strip(" - "),
                    "quantity": str(comp.quantity),
                    "designator": comp.designator or ""
                }
                parts_data.append(part)

            result = await self.agent.container.services.bom.add_parts(bom_name.strip(), "", parts_data)

            if result.get('success'):
                enhanced_count = sum(1 for comp in components if comp.enhanced)

                return AgentResponse(
                    success=True,
                    action=self.name,
                    message=f"Successfully added {len(components)} components to BOM '{bom_name}' ({enhanced_count} enhanced).",
                    data={
                        "bom_name": bom_name,
                        "components_added": len(components),
                        "enhanced_count": enhanced_count
                    }
                )
            else:
                return AgentResponse(
                    success=False,
                    action=self.name,
                    error=result.get('error', 'Unknown error'),
                    message=f"Failed to add components to BOM '{bom_name}'"
                )

        except Exception as e:
            return AgentResponse(
                success=False,
                action=self.name,
                error=str(e),
                message=f"Adding components to BOM failed: {str(e)}"
            )


class ToolRegistry:
    """Dynamic tool registry with proper LangChain integration"""

    def __init__(self, agent: 'BOMAgent'):
        self.agent = agent
        self.tools: Dict[str, AgentTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""
        self.register_tool(SchematicAnalysisTool)
        self.register_tool(ComponentSearchTool)
        self.register_tool(BOMCreateTool)
        self.register_tool(BOMAddComponentsTool)

    def register_tool(self, tool_class: type):
        """Register a new tool dynamically"""
        tool_instance = tool_class(self.agent)
        self.tools[tool_instance.name] = tool_instance

    def get_tools_for_llm(self) -> List:
        """Get tools formatted for LangChain with proper schemas"""
        llm_tools = []

        for tool_name, tool_instance in self.tools.items():
            @tool(
                name_or_callable=tool_name,
                description=tool_instance.description,
                args_schema=tool_instance.args_schema
            )
            async def tool_wrapper(**kwargs):
                # This creates a closure that captures the tool_instance
                return await tool_instance.execute(**kwargs)

            # Store reference to original tool for execution
            tool_wrapper._tool_instance = tool_instance
            llm_tools.append(tool_wrapper)

        return llm_tools

    async def execute_tool(self, tool_name: str, **kwargs) -> AgentResponse:
        """Execute a tool by name"""
        if tool_name in self.tools:
            return await self.tools[tool_name].execute(**kwargs)
        else:
            return AgentResponse(
                success=False,
                action=tool_name,
                error=f"Tool '{tool_name}' not found"
            )


# =============================================================================
# Agent State for LangGraph
# =============================================================================

class AgentState(BaseModel):
    """Clean agent state"""
    messages: List[Any] = []
    session_id: str
    stored_components: List[Dict[str, Any]] = []
    last_response: Optional[Dict[str, Any]] = None


# =============================================================================
# Main Agent - Optimized and Clean
# =============================================================================

class BOMAgent:
    """Optimized, clean BOM agent with proper ReAct implementation"""

    def __init__(self, config, session_id: str = None):
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())

        # Initialize Container
        self.container = Container(config, session_id)

        # Component storage
        self._stored_components: List[ComponentData] = []
        self.conversation_history: List[Dict[str, Any]] = []

        # Tool registry
        self.tool_registry = None

        # LangGraph
        self.graph = None

        # Component state manager
        self.component_state = ComponentStateManager()

    async def initialize(self) -> None:
        """Initialize all services"""
        try:
            await self.container.initialize()

            # Initialize tool registry
            self.tool_registry = ToolRegistry(self)

            # Build LangGraph workflow
            self._build_graph()

            print(f"✅ Optimized Agent initialized successfully for session {self.session_id}")

        except Exception as e:
            print(f"❌ Agent initialization failed: {str(e)}")
            raise

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
        system_msg = SystemMessage(content=f"""You are a professional electronic design BOM assistant with advanced schematic analysis capabilities.

**Your Capabilities:**
- analyze_schematic: Analyze electronic schematics and extract components with Silicon Expert enhancement
- search_component: Search for component specifications in Silicon Expert database
- create_bom: Create and manage Bills of Materials
- add_components_to_bom: Add analyzed components to existing BOMs

**Current Context:**
- Session: {state.session_id}
- Stored Components: {len(state.stored_components)} available

**Instructions:**
1. For schematic analysis: Use analyze_schematic with the exact image URL
2. Always use proper tool parameter names (image_url, query, name, bom_name)
3. Chain tools when appropriate (analyze → create_bom → add_components_to_bom)
4. Provide helpful and structured responses
5. When users provide URLs, use them directly in the analyze_schematic tool

**Tool Parameters:**
- analyze_schematic(image_url="https://...")
- search_component(query="component_name") 
- create_bom(name="BOM_Name", project="Project", description="Description")
- add_components_to_bom(bom_name="BOM_Name")

Focus on understanding user intent and providing accurate technical assistance.""")

        # Prepare messages
        messages = state.messages.copy()
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, system_msg)

        # Get LLM response with tools
        agent = self.container.get_agent_llm().bind_tools(self.tool_registry.get_tools_for_llm())
        response = await agent.ainvoke(messages)

        # Update state
        new_state = state.model_copy()
        new_state.messages = messages + [response]

        return new_state

    async def _tool_node(self, state: AgentState) -> AgentState:
        """Execute tool calls with proper error handling"""
        last_message = state.messages[-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            results = []

            for tool_call in last_message.tool_calls:
                result = await self._execute_tool_call(tool_call)
                results.append(result)

                # Update stored components if this was a schematic analysis
                if (tool_call['name'] == 'analyze_schematic' and
                        result.get('success') and
                        result.get('data', {}).get('components')):
                    components_data = result['data']['components']
                    self.component_state.store_raw_analysis(
                        [ComponentData.from_dict(comp_data) for comp_data in components_data])

            new_state = state.model_copy()
            new_state.last_response = results[-1] if results else None

            # Add tool results to messages
            for result in results:
                # Create a summary message instead of dumping all data
                if result.get('success'):
                    if result.get('action') == 'analyze_schematic':
                        # For schematic analysis, just summarize
                        component_count = len(result.get('data', {}).get('components', []))
                        enhanced_count = result.get('data', {}).get('enhanced_count', 0)
                        summary_message = f"✅ Schematic analysis completed successfully. Found {component_count} components, enhanced {enhanced_count} with Silicon Expert data."
                    else:
                        # For other tools, use the message field
                        summary_message = result.get('message', 'Tool executed successfully')
                else:
                    summary_message = f"❌ Tool execution failed: {result.get('error', 'Unknown error')}"

                # Add concise summary instead of full data
                new_state.messages.append(AIMessage(content=summary_message))

            return new_state

        return state

    async def _execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute a specific tool call with proper parameter handling"""
        tool_name = tool_call['name']
        tool_args = tool_call.get('args', {})

        try:
            # Get the tool instance
            if tool_name in self.tool_registry.tools:
                tool_instance = self.tool_registry.tools[tool_name]

                # Execute with proper parameter unpacking
                result = await tool_instance.execute(**tool_args)
                return result.to_dict()
            else:
                return AgentResponse(
                    success=False,
                    action=tool_name,
                    error=f"Tool '{tool_name}' not found"
                ).to_dict()

        except Exception as e:
            return AgentResponse(
                success=False,
                action=tool_name,
                error=str(e),
                message=f"Tool execution failed: {str(e)}"
            ).to_dict()

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if workflow should continue"""
        last_message = state.messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"

    # =============================================================================
    # Public Interface Methods
    # =============================================================================

    async def process_request(self, message: str) -> AgentResponse:
        """Process user request and return structured response"""
        try:
            # Create initial state
            initial_state = AgentState(
                messages=[HumanMessage(content=message)],
                session_id=self.session_id,
                stored_components=[comp.to_dict() for comp in self._stored_components]
            )

            # Run workflow
            config = {"configurable": {"thread_id": self.session_id}}
            final_state = await self.graph.ainvoke(initial_state, config)

            # Update stored components from component_state manager
            stored_components = self.component_state.get_components_for_bom()
            self._stored_components = stored_components

            # Add to conversation history
            self.conversation_history.append({
                "type": "human",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })

            # Extract and return final response
            if final_state.get('last_response'):
                response_data = final_state['last_response']

                # Store in conversation history
                self.conversation_history.append({
                    "type": "assistant",
                    "content": response_data,
                    "timestamp": datetime.now().isoformat()
                })

                # Return the structured response
                return AgentResponse(**response_data)
            else:
                # Handle case where no tool was executed
                # Look for the last AI message
                last_ai_message = None
                for msg in reversed(final_state.get('messages', [])):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break

                if last_ai_message:
                    response_message = last_ai_message.content
                else:
                    response_message = "I'm here to help with electronic design and BOM management. You can ask me to analyze schematics, search for components, or manage BOMs."

                # Create a general response
                response = AgentResponse(
                    success=True,
                    action="general_response",
                    message=response_message
                )

                self.conversation_history.append({
                    "type": "assistant",
                    "content": response.to_dict(),
                    "timestamp": datetime.now().isoformat()
                })

                return response

        except Exception as e:
            print(f"❌ Request processing error: {str(e)}")
            error_response = AgentResponse(
                success=False,
                action="error",
                error=str(e),
                message=f"Request processing failed: {str(e)}"
            )

            self.conversation_history.append({
                "type": "assistant",
                "content": error_response.to_dict(),
                "timestamp": datetime.now().isoformat()
            })

            return error_response

    async def get_stored_components(self) -> List[ComponentData]:
        """Get stored components"""
        return self._stored_components.copy()

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

    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.container.cleanup()

    # =============================================================================
    # Dynamic Tool Management
    # =============================================================================

    def add_custom_tool(self, tool_class: type):
        """Add a custom tool dynamically"""
        if self.tool_registry:
            self.tool_registry.register_tool(tool_class)
            # Rebuild the graph to include new tools
            self._build_graph()

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        if self.tool_registry:
            return list(self.tool_registry.tools.keys())
        return []

    @property
    def stored_components(self):
        return self._stored_components


# =============================================================================
# Example Custom Tool - Shows Extensibility
# =============================================================================

class ComponentListInput(BaseModel):
    """Input schema for component list tool"""
    format: str = Field(default="table", description="Display format: table, json, or csv")


class ComponentListTool(AgentTool):
    """Example custom tool for listing stored components"""

    @property
    def name(self) -> str:
        return "list_components"

    @property
    def description(self) -> str:
        return """List all stored components in specified format.

        Input: format (string, optional) - Display format: "table", "json", or "csv"
        Output: Formatted list of all stored components

        Example: list_components(format="table")"""

    @property
    def args_schema(self):
        return ComponentListInput

    async def execute(self, resp_format: str = "table") -> AgentResponse:
        """List components in specified format"""
        components = self.agent.component_state.get_components_for_bom()

        if not components:
            return AgentResponse(
                success=False,
                action=self.name,
                error="No components available",
                message="No components found. Please analyze a schematic first."
            )

        try:
            if resp_format.lower() == "json":
                components_data = [comp.to_dict() for comp in components]
                return AgentResponse(
                    success=True,
                    action=self.name,
                    message=f"Listed {len(components)} components in JSON format",
                    data={"components": components_data, "format": "json"}
                )

            elif resp_format.lower() == "csv":
                # Create CSV-like data structure
                csv_data = []
                for comp in components:
                    csv_data.append({
                        "Name": comp.name,
                        "Part Number": comp.part_number or "",
                        "Manufacturer": comp.manufacturer or "",
                        "Value": comp.value or "",
                        "Designator": comp.designator or "",
                        "Enhanced": "Yes" if comp.enhanced else "No"
                    })

                return AgentResponse(
                    success=True,
                    action=self.name,
                    message=f"Listed {len(components)} components in CSV format",
                    data={"components": csv_data, "format": "csv"}
                )

            else:  # Default table format
                table_data = []
                for i, comp in enumerate(components, 1):
                    table_data.append({
                        "No": i,
                        "Name": comp.name,
                        "Part Number": comp.part_number or "N/A",
                        "Manufacturer": comp.manufacturer or "N/A",
                        "Value": comp.value or "N/A",
                        "Designator": comp.designator or f"COMP{i}",
                        "Enhanced": "✅" if comp.enhanced else "📋"
                    })

                return AgentResponse(
                    success=True,
                    action=self.name,
                    message=f"Listed {len(components)} components in table format",
                    data={"components": table_data, "format": "table"}
                )

        except Exception as e:
            return AgentResponse(
                success=False,
                action=self.name,
                error=str(e),
                message=f"Failed to list components: {str(e)}"
            )


# =============================================================================
# Agent Factory and Alias for Backward Compatibility
# =============================================================================

class BOMAgentFactory:
    """Factory for creating different agent configurations"""

    @staticmethod
    async def create_standard_agent(config, session_id: str = None) -> BOMAgent:
        """Create standard agent with default tools"""
        agent = BOMAgent(config, session_id)
        await agent.initialize()
        return agent

    @staticmethod
    async def create_extended_agent(config, session_id: str = None) -> BOMAgent:
        """Create agent with additional tools"""
        agent = BOMAgent(config, session_id)
        await agent.initialize()

        # Add custom tools
        agent.add_custom_tool(ComponentListTool)

        return agent


# =============================================================================
# Usage Example and Testing
# =============================================================================

async def test_agent():
    """Test the optimized agent"""
    from config import AppConfig

    # Load configuration
    config = AppConfig.from_env()

    # Create and initialize agent
    agent = await BOMAgentFactory.create_extended_agent(config)

    print("🤖 Optimized BOM Agent ready!")
    print(f"📋 Available tools: {', '.join(agent.get_available_tools())}")

    # Test the problematic URL from your example
    test_url = "https://www.tronicszone.com/tronicszone/wp-content/uploads/2020/09/circuit-design-tips-1030x764.png"

    try:
        print(f"\n🔍 Testing schematic analysis with: {test_url}")
        response = await agent.process_request(f"analyze schematic at {test_url}")

        print(f"✅ Response: {response.message}")
        if response.data:
            components = response.data.get('components', [])
            print(f"📦 Found {len(components)} components")

        # Test other operations
        if response.success:
            print("\n🔍 Testing component listing...")
            list_response = await agent.process_request("list all components in table format")
            print(f"📋 List response: {list_response.message}")

            print("\n🔍 Testing BOM creation...")
            bom_response = await agent.process_request("create BOM named 'Test Circuit' for project 'Demo'")
            print(f"📋 BOM response: {bom_response.message}")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Cleanup
        await agent.cleanup()
        print("✅ Agent cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_agent())