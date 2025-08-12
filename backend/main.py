# main.py - Simplified React-based BOM Agent
"""
Clean, maintainable BOM Agent using React pattern.
Designed for Angular chat UI integration via REST API.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

from langchain.agents import Tool, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI


# ================================================================================================
# CORE MODELS & TYPES
# ================================================================================================

class ResponseType(Enum):
    """UI response types for Angular frontend"""
    SUCCESS = "success"
    ERROR = "error"
    TABLE = "table"
    CONFIRMATION = "confirmation"
    PROGRESS = "progress"
    TREE = "tree"


@dataclass
class AgentResponse:
    """Standardized response for Angular UI"""
    id: str
    success: bool
    type: ResponseType
    message: str
    data: Optional[Dict[str, Any]] = None
    suggestions: List[str] = None
    timestamp: str = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.suggestions is None:
            self.suggestions = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComponentData:
    """Simplified component data structure"""
    name: str
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    quantity: int = 1
    confidence: float = 0.0


# ================================================================================================
# ABSTRACT SERVICE INTERFACES
# ================================================================================================

class ServiceInterface(ABC):
    """Base interface for all services"""

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass


class SchematicServiceInterface(ServiceInterface):
    @abstractmethod
    async def analyze_schematic(self, image_url: str) -> Dict[str, Any]:
        pass


class ComponentServiceInterface(ServiceInterface):
    @abstractmethod
    async def search_components(self, query: str) -> Dict[str, Any]:
        pass


class BOMServiceInterface(ServiceInterface):
    @abstractmethod
    async def create_bom(self, name: str, project: str = "", description: str = "") -> Dict[str, Any]:
        pass

    @abstractmethod
    async def add_parts_to_bom(self, bom_name: str, parts: List[ComponentData]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def list_boms(self, project_filter: str = "") -> Dict[str, Any]:
        pass


# ================================================================================================
# SIMPLIFIED SERVICE IMPLEMENTATIONS
# ================================================================================================

class SchematicService(SchematicServiceInterface):
    """Wrapper for existing schematic service"""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self._initialized = False
        self._original_service = None

    async def initialize(self) -> None:
        if not self._initialized:
            from src.services.schematic_service import SchematicService as OriginalService
            self._original_service = OriginalService(self.llm)
            self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False
        self._original_service = None

    async def analyze_schematic(self, image_url: str) -> Dict[str, Any]:
        """Analyze schematic and return standardized result"""
        if not self._initialized:
            await self.initialize()

        try:
            result = await self._original_service.analyze(image_url)
            components = result.get("components", [])

            # Store components in session memory for later use
            self._last_components = [
                ComponentData(
                    name=c.get('name', 'Unknown'),
                    part_number=c.get('part_number'),
                    manufacturer=c.get('manufacturer'),
                    description=c.get('description', ''),
                    confidence=c.get('confidence', 0.0)
                ) for c in components
            ]

            return {
                "success": True,
                "components": components,
                "total_found": len(components),
                "confidence": sum(c.get("confidence", 0) for c in components) / max(1, len(components)),
                "image_url": image_url
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Schematic analysis failed: {str(e)}",
                "components": []
            }

    def get_last_components(self) -> List[ComponentData]:
        """Get components from last analysis"""
        return getattr(self, '_last_components', [])


class ComponentService(ComponentServiceInterface):
    """Wrapper for existing component service"""

    def __init__(self, silicon_expert_client, memory_service):
        self.client = silicon_expert_client
        self.memory = memory_service
        self._initialized = False
        self._original_service = None

    async def initialize(self) -> None:
        if not self._initialized:
            from src.services.component_service import ComponentService as OriginalService
            self._original_service = OriginalService(self.client, self.memory)
            self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False
        self._original_service = None

    async def search_components(self, query: str) -> Dict[str, Any]:
        """Search for components using existing service"""
        if not self._initialized:
            await self.initialize()

        try:
            # Convert query to component search format
            components = [{"name": query, "description": query}]
            enhanced = await self._original_service.search_and_enhance(components)

            if enhanced:
                comp = enhanced[0]
                return {
                    "success": True,
                    "components": [{
                        "name": comp.name,
                        "part_number": comp.part_number,
                        "manufacturer": comp.manufacturer,
                        "description": comp.description,
                        "confidence": comp.confidence
                    }],
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "error": f"No components found for: {query}",
                    "components": []
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Component search failed: {str(e)}",
                "components": []
            }


class BOMService(BOMServiceInterface):
    """Wrapper for existing BOM service"""

    def __init__(self, silicon_expert_client, memory_service):
        self.client = silicon_expert_client
        self.memory = memory_service
        self._initialized = False
        self._original_service = None

    async def initialize(self) -> None:
        if not self._initialized:
            from src.services.bom_service import BOMService as OriginalService
            self._original_service = OriginalService(self.client, self.memory)
            self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False
        self._original_service = None

    async def create_bom(self, name: str, project: str = "", description: str = "") -> Dict[str, Any]:
        """Create new BOM using existing service"""
        if not self._initialized:
            await self.initialize()

        try:
            result = await self._original_service.create_bom(name, description, project)

            return {
                "success": result.get("success", False),
                "bom_name": name,
                "project": project,
                "message": f"BOM '{name}' created successfully" if result.get("success") else result.get("error",
                                                                                                         "Failed to create BOM")
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"BOM creation failed: {str(e)}"
            }

    async def add_parts_to_bom(self, bom_name: str, parts: List[ComponentData]) -> Dict[str, Any]:
        """Add parts to existing BOM"""
        if not self._initialized:
            await self.initialize()

        try:
            # Convert ComponentData to API format
            api_parts = []
            for part in parts:
                api_parts.append({
                    "part_number": part.part_number or "Unknown",
                    "manufacturer": part.manufacturer or "Unknown",
                    "description": part.description or part.name,
                    "quantity": part.quantity,
                    "designator": ""
                })

            parts_data = {
                "bom_name": bom_name,
                "project": "",
                "parts": api_parts
            }

            result = await self._original_service.add_parts(bom_name, "", api_parts)

            return {
                "success": result.get("success", False),
                "bom_name": bom_name,
                "parts_added": len(parts),
                "message": f"Added {len(parts)} parts to BOM '{bom_name}'" if result.get("success") else result.get(
                    "error", "Failed to add parts")
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to add parts: {str(e)}"
            }

    async def list_boms(self, project_filter: str = "") -> Dict[str, Any]:
        """List existing BOMs"""
        if not self._initialized:
            await self.initialize()

        try:
            result = await self._original_service.get_boms(project_filter)

            return {
                "success": result.get("success", False),
                "boms": result.get("boms", []),
                "total_count": len(result.get("boms", [])),
                "message": f"Found {len(result.get('boms', []))} BOMs"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list BOMs: {str(e)}",
                "boms": []
            }


# ================================================================================================
# SIMPLIFIED SERVICE CONTAINER
# ================================================================================================

class ServiceContainer:
    """Simplified dependency injection container"""

    def __init__(self, config, session_id: str = None):
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self._services: Dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all services"""
        if self._initialized:
            return

        try:
            # Create LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=self.config.google_api_key,
                temperature=0.1,
                max_output_tokens=30000,
            )

            # Create Silicon Expert client
            from src.clients.silicon_expert_client import SiliconExpertClient
            se_client = SiliconExpertClient(self.config.silicon_expert)
            await se_client.authenticate()

            # Create memory service
            from src.services.memory_service import MemoryService
            memory_service = MemoryService(self.session_id)

            # Initialize services
            self._services = {
                "llm": llm,
                "silicon_expert_client": se_client,
                "memory": memory_service,
                "schematic": SchematicService(llm),
                "component": ComponentService(se_client, memory_service),
                "bom": BOMService(se_client, memory_service)
            }

            # Initialize all services
            for name, service in self._services.items():
                if isinstance(service, ServiceInterface):
                    await service.initialize()

            self._initialized = True

        except Exception as e:
            print(f"Service container initialization failed: {str(e)}")
            raise

    def get_service(self, service_name: str) -> Any:
        """Get service by name"""
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call await container.initialize() first.")
        return self._services.get(service_name)

    async def cleanup(self) -> None:
        """Cleanup all services"""
        for service in self._services.values():
            if isinstance(service, ServiceInterface):
                await service.cleanup()
            elif hasattr(service, 'close'):
                await service.close()

        self._services.clear()
        self._initialized = False


# ================================================================================================
# TOOL FACTORY - Creates React Agent Tools
# ================================================================================================

class ToolFactory:
    """Factory for creating React agent tools"""

    def __init__(self, container: ServiceContainer):
        self.container = container

    def create_tools(self) -> List[Tool]:
        """Create all BOM agent tools"""
        return [
            self._create_analyze_schematic_tool(),
            self._create_search_components_tool(),
            self._create_create_bom_tool(),
            self._create_add_parts_tool(),
            self._create_list_boms_tool(),
            self._create_help_tool()
        ]

    def _create_analyze_schematic_tool(self) -> Tool:
        async def analyze_schematic(image_url: str) -> str:
            service = self.container.get_service("schematic")
            result = await service.analyze_schematic(image_url)

            if result["success"]:
                components = result["components"]
                confidence = result.get("confidence", 0)

                if not components:
                    return "⚠️ No components found in the schematic. Please check the image quality and try again."

                output = f"✅ Analyzed schematic successfully!\n"
                output += f"📊 Found {len(components)} components (Average confidence: {confidence:.1%})\n\n"

                # Show top 5 components with details
                for i, comp in enumerate(components[:5], 1):
                    name = comp.get('name', 'Unknown')
                    conf = comp.get('confidence', 0)
                    desc = comp.get('description', '')
                    output += f"{i}. {name} ({conf:.1%} confidence)"
                    if desc:
                        output += f" - {desc}"
                    output += "\n"

                if len(components) > 5:
                    output += f"... and {len(components) - 5} more components\n"

                output += "\n💡 Next steps: Create a BOM with these components or search for more details."
                return output
            else:
                return f"❌ Schematic analysis failed: {result['error']}"

        return Tool(
            name="analyze_schematic",
            description="Analyze circuit schematic from image URL to extract components. Input should be a valid image URL (http/https).",
            func=lambda url: asyncio.run(analyze_schematic(url))
        )

    def _create_search_components_tool(self) -> Tool:
        async def search_components(query: str) -> str:
            service = self.container.get_service("component")
            result = await service.search_components(query)

            if result["success"] and result["components"]:
                comp = result["components"][0]
                output = f"✅ Component found!\n\n"
                output += f"🔍 **Search Result for: {query}**\n"
                output += f"📦 Part Number: {comp.get('part_number') or 'Not available'}\n"
                output += f"🏭 Manufacturer: {comp.get('manufacturer') or 'Not available'}\n"
                output += f"📝 Description: {comp.get('description') or 'Not available'}\n"
                output += f"⭐ Confidence: {comp.get('confidence', 0):.1%}\n\n"
                output += "💡 You can now add this component to a BOM or search for similar parts."
                return output
            else:
                return f"❌ No components found matching '{query}'. Try using:\n" + \
                    "• Part numbers (e.g., 'LM358')\n" + \
                    "• Component types (e.g., 'Arduino Uno')\n" + \
                    "• Manufacturer codes"

        return Tool(
            name="search_components",
            description="Search for electronic components by name, part number, or description. Input should be component name, part number, or description.",
            func=lambda query: asyncio.run(search_components(query))
        )

    def _create_create_bom_tool(self) -> Tool:
        async def create_bom(params: str) -> str:
            # Parse parameters: "name=MyBOM,project=Arduino,description=Power supply BOM"
            parsed = self._parse_tool_params(params)
            name = parsed.get("name", "").strip()

            if not name:
                return "❌ BOM name is required.\n\n" + \
                    "📝 **Usage:** name=BOM_NAME,project=PROJECT_NAME,description=DESCRIPTION\n" + \
                    "📋 **Example:** name=PowerSupply,project=Arduino,description=Main power circuit"

            service = self.container.get_service("bom")
            result = await service.create_bom(
                name=name,
                project=parsed.get("project", ""),
                description=parsed.get("description", "")
            )

            if result["success"]:
                output = f"✅ BOM created successfully!\n\n"
                output += f"📋 **BOM Details:**\n"
                output += f"• Name: {name}\n"
                if parsed.get("project"):
                    output += f"• Project: {parsed.get('project')}\n"
                if parsed.get("description"):
                    output += f"• Description: {parsed.get('description')}\n"
                output += "\n💡 Next step: Add components to your new BOM."
                return output
            else:
                return f"❌ Failed to create BOM: {result['error']}"

        return Tool(
            name="create_bom",
            description="Create a new Bill of Materials (BOM). Input format: name=BOM_NAME,project=PROJECT_NAME,description=DESCRIPTION (project and description are optional)",
            func=lambda params: asyncio.run(create_bom(params))
        )

    def _create_add_parts_tool(self) -> Tool:
        async def add_parts(params: str) -> str:
            parsed = self._parse_tool_params(params)
            bom_name = parsed.get("bom_name", "").strip()

            if not bom_name:
                return "❌ BOM name is required.\n\n" + \
                    "📝 **Usage:** bom_name=BOM_NAME\n" + \
                    "⚠️ **Note:** Components must be found first (via schematic analysis or search)"

            # Get components from schematic service
            schematic_service = self.container.get_service("schematic")
            components = schematic_service.get_last_components() if hasattr(schematic_service,
                                                                            'get_last_components') else []

            if not components:
                return "⚠️ No components available to add.\n\n" + \
                    "💡 **To add parts:**\n" + \
                    "1. First analyze a schematic or search for components\n" + \
                    "2. Then use this command to add them to your BOM"

            bom_service = self.container.get_service("bom")
            result = await bom_service.add_parts_to_bom(bom_name, components)

            if result["success"]:
                output = f"✅ Parts added successfully!\n\n"
                output += f"📋 **Added to BOM:** {bom_name}\n"
                output += f"📦 **Parts Count:** {len(components)}\n\n"
                output += "**Components Added:**\n"
                for i, comp in enumerate(components[:5], 1):
                    output += f"{i}. {comp.name}"
                    if comp.part_number:
                        output += f" ({comp.part_number})"
                    output += "\n"
                if len(components) > 5:
                    output += f"... and {len(components) - 5} more\n"
                return output
            else:
                return f"❌ Failed to add parts: {result['error']}"

        return Tool(
            name="add_parts_to_bom",
            description="Add previously analyzed/searched components to an existing BOM. Input format: bom_name=BOM_NAME",
            func=lambda params: asyncio.run(add_parts(params))
        )

    def _create_list_boms_tool(self) -> Tool:
        async def list_boms(project_filter: str = "") -> str:
            service = self.container.get_service("bom")
            result = await service.list_boms(project_filter.strip())

            if result["success"]:
                boms = result["boms"]
                if not boms:
                    return "📝 No BOMs found.\n\n💡 Create your first BOM to get started!"

                output = f"📋 **BOMs Overview** ({len(boms)} found)\n\n"

                # Group by project if available
                projects = {}
                for bom in boms:
                    project = bom.get('project') or 'Unassigned'
                    if project not in projects:
                        projects[project] = []
                    projects[project].append(bom)

                for project, project_boms in projects.items():
                    if len(projects) > 1:
                        output += f"**📁 {project}:**\n"

                    for bom in project_boms[:10]:  # Limit display
                        name = bom.get('name', 'Unknown')
                        created = bom.get('created_at', '')
                        parts_count = bom.get('component_count', 'N/A')

                        output += f"• {name}"
                        if parts_count != 'N/A':
                            output += f" ({parts_count} parts)"
                        if created:
                            try:
                                from datetime import datetime
                                dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                                output += f" - Created {dt.strftime('%Y-%m-%d')}"
                            except:
                                pass
                        output += "\n"

                    if len(project_boms) > 10:
                        output += f"... and {len(project_boms) - 10} more BOMs\n"
                    output += "\n"

                return output.strip()
            else:
                return f"❌ Failed to retrieve BOMs: {result['error']}"

        return Tool(
            name="list_boms",
            description="List all existing BOMs, optionally filtered by project name. Input: project_name (optional, leave empty for all BOMs)",
            func=lambda project: asyncio.run(list_boms(project))
        )

    def _create_help_tool(self) -> Tool:
        def show_help(_: str = "") -> str:
            return """
🔧 **BOM Agent - Your Electronic Design Assistant**

**🎯 Core Capabilities:**
1. **📸 Analyze Schematics** - Extract components from circuit diagrams
2. **🔍 Search Components** - Find electronic parts and specifications  
3. **📋 Create BOMs** - Build Bill of Materials for projects
4. **📦 Add Parts** - Add components to existing BOMs
5. **📊 List BOMs** - View and manage your BOMs

**💬 Example Commands:**
• *"Analyze this schematic: https://example.com/circuit.jpg"*
• *"Search for Arduino Uno microcontroller"*
• *"Create BOM called PowerSupply for Arduino project with description Main power circuit"*
• *"Show me all my BOMs"*
• *"Add parts to PowerSupply BOM"*

**🏗️ Typical Workflow:**
1. **Analyze** a schematic to extract components
2. **Create** a new BOM for your project  
3. **Add** the analyzed components to your BOM
4. **Search** for additional components as needed
5. **List** your BOMs to manage projects

**💡 Pro Tips:**
• Use clear, high-quality schematic images for best results
• Include project names to organize your BOMs
• Search using specific part numbers for accurate results

Ready to help with your electronic design projects! 🚀
            """

        return Tool(
            name="help",
            description="Show available commands, capabilities, and usage examples",
            func=show_help
        )

    def _parse_tool_params(self, params: str) -> Dict[str, str]:
        """Parse comma-separated key=value parameters"""
        result = {}
        if not params or params.strip() == "":
            return result

        try:
            for pair in params.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    result[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error parsing tool parameters '{params}': {e}")

        return result


# ================================================================================================
# MAIN SIMPLIFIED BOM AGENT
# ================================================================================================

class SimpleBOMAgent:
    """Simplified React-based BOM Agent for Angular UI integration"""

    def __init__(self, config, session_id: Optional[str] = None):
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self.container = ServiceContainer(config, self.session_id)
        self.agent = None
        self._conversation_history: List[Dict[str, Any]] = []
        self._max_history = 50  # Limit conversation history

    async def initialize(self) -> None:
        """Initialize the agent and all services"""
        try:
            await self.container.initialize()

            # Create tools and agent
            tool_factory = ToolFactory(self.container)
            tools = tool_factory.create_tools()

            llm = self.container.get_service("llm")

            self.agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.SELF_ASK_WITH_SEARCH,
                verbose=False,
                max_iterations=3,  # Limit iterations for faster response
                early_stopping_method="generate",
                return_intermediate_steps=False  # Cleaner output
            )

        except Exception as e:
            print(f"Agent initialization failed: {str(e)}")
            raise

    async def process_request(self, user_input: str) -> AgentResponse:
        """Process user request and return standardized response for Angular UI"""
        response_id = str(uuid.uuid4())

        try:
            if not self.agent:
                await self.initialize()

            # Add to conversation history
            self._add_to_history("human", user_input)

            # Process with React agent
            result = await self.agent.arun(user_input)

            # Clean up the result (remove any intermediate steps formatting)
            cleaned_result = self._clean_agent_output(result)

            # Add agent response to history
            self._add_to_history("ai", cleaned_result)

            # Determine response type and suggestions
            response_type = self._determine_response_type(cleaned_result)
            suggestions = self._generate_suggestions(user_input, cleaned_result)

            return AgentResponse(
                id=response_id,
                success=True,
                type=response_type,
                message=cleaned_result,
                data={"session_id": self.session_id},
                suggestions=suggestions
            )

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error processing your request: {str(e)}\n\n" + \
                        "Please try:\n• Rephrasing your request\n• Using the 'help' command\n• Checking if URLs are accessible"

            self._add_to_history("ai", error_msg)

            return AgentResponse(
                id=response_id,
                success=False,
                type=ResponseType.ERROR,
                message=error_msg,
                suggestions=["Try rephrasing your request", "Use 'help' to see available commands",
                             "Check your input format"]
            )

    def _clean_agent_output(self, output: str) -> str:
        """Clean up agent output for better UI presentation"""
        # Remove any React agent formatting artifacts
        cleaned = output.replace("Action:", "").replace("Observation:", "")
        cleaned = cleaned.replace("Thought:", "").replace("Final Answer:", "")

        # Remove excessive newlines
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")

        return cleaned.strip()

    def _add_to_history(self, message_type: str, content: str):
        """Add message to conversation history with size limit"""
        self._conversation_history.append({
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only recent messages
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

    def _determine_response_type(self, result: str) -> ResponseType:
        """Determine UI response type based on agent output"""
        result_lower = result.lower()

        if "❌" in result or "error" in result_lower or "failed" in result_lower or "apologize" in result_lower:
            return ResponseType.ERROR
        elif ("found" in result_lower and ("component" in result_lower or "bom" in result_lower)) or "📋" in result:
            return ResponseType.TABLE
        elif "📁" in result or "project" in result_lower:
            return ResponseType.TREE
        elif "✅" in result or "successfully" in result_lower:
            return ResponseType.SUCCESS
        else:
            return ResponseType.SUCCESS

    def _generate_suggestions(self, user_input: str, agent_output: str) -> List[str]:
        """Generate contextual suggestions for Angular UI"""
        suggestions = []
        user_lower = user_input.lower()
        output_lower = agent_output.lower()

        # Context-aware suggestions
        if "schematic" in user_lower and ("found" in output_lower or "✅" in agent_output):
            suggestions.extend([
                "Create a new BOM with these components",
                "Search for more component details",
                "Add these parts to existing BOM"
            ])
        elif "search" in user_lower and ("found" in output_lower or "✅" in agent_output):
            suggestions.extend([
                "Add this component to a BOM",
                "Search for similar components",
                "Create new BOM with this part"
            ])
        elif "create" in user_lower and "bom" in user_lower and "✅" in agent_output:
            suggestions.extend([
                "Add components to this BOM",
                "Analyze a schematic for parts",
                "List all BOMs"
            ])
        elif "list" in user_lower or "show" in user_lower:
            suggestions.extend([
                "Create a new BOM",
                "Analyze a schematic",
                "Search for components"
            ])
        elif "❌" in agent_output or "error" in output_lower:
            suggestions.extend([
                "Try rephrasing your request",
                "Use 'help' command",
                "Check input format"
            ])
        else:
            suggestions.extend([
                "Analyze a schematic image",
                "Search for components",
                "Create a new BOM",
                "Show help"
            ])

        return suggestions[:3]  # Limit to 3 suggestions

    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for Angular UI"""
        return self._conversation_history.copy()

    async def clear_history(self) -> None:
        """Clear conversation history"""
        self._conversation_history.clear()

    async def get_session_info(self) -> Dict[str, Any]:
        """Get session information"""
        return {
            "session_id": self.session_id,
            "history_count": len(self._conversation_history),
            "initialized": self.agent is not None,
            "last_activity": self._conversation_history[-1]["timestamp"] if self._conversation_history else None
        }

    async def cleanup(self) -> None:
        """Cleanup resources"""
        await self.container.cleanup()
        self._conversation_history.clear()
        self.agent = None
