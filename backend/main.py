# main.py - MODERN LANGGRAPH AGENT
"""Modern LangGraph agent with clean architecture and human-in-the-loop patterns."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from src.core.config import AppConfig
from src.core.container import Container
from src.models.state import AgentState, ResponseType, HumanApprovalRequest, AgentResponse
from src.models.state import UIRecommendation
from src.services.agent_service import AgentOrchestrator


class WorkflowStep(Enum):
    """Workflow step enumeration."""
    ANALYZE_SCHEMATIC = "analyze_schematic"
    SEARCH_COMPONENTS = "search_components"
    CREATE_BOM = "create_bom"
    ADD_PARTS = "add_parts"
    HUMAN_APPROVAL = "human_approval"


@dataclass
class IntentResult:
    """Intent classification result."""
    intent: str
    confidence: float
    parameters: Dict[str, Any]
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class HumanApprovalConfig:
    """Configuration for human approval requirements."""
    require_analysis_approval: bool = True
    require_bom_creation_approval: bool = True
    require_parts_addition_approval: bool = True
    auto_approve_threshold: Optional[float] = 0.95


class ModernBOMAgent:
    """Fixed LangGraph agent with proper routing and error handling."""

    def __init__(self, config: AppConfig, session_id: Optional[str] = None):
        self.config = config
        self.container = Container(config, session_id)
        self.orchestrator = AgentOrchestrator(self.container)
        self.approval_config = HumanApprovalConfig()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the corrected LangGraph workflow with BOM listing."""
        workflow = StateGraph(AgentState)

        # Add all nodes including new BOM listing nodes
        workflow.add_node("router", self._route_request)
        workflow.add_node("validate_schematic_input", self._validate_schematic_input)
        workflow.add_node("validate_search_input", self._validate_search_input)
        workflow.add_node("validate_bom_input", self._validate_bom_input)
        workflow.add_node("validate_parts_input", self._validate_parts_input)
        workflow.add_node("validate_bom_list_input", self._validate_bom_list_input)  # New
        workflow.add_node("analyze_schematic", self._analyze_schematic)
        workflow.add_node("search_components", self._search_components)
        workflow.add_node("create_bom", self._create_bom)
        workflow.add_node("add_parts", self._add_parts)
        workflow.add_node("list_boms", self._list_boms)  # New
        workflow.add_node("human_approval", self._request_human_approval)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("handle_retry", self._handle_retry)
        workflow.add_node("format_response", self._format_response)

        # Fixed conditional routing from router
        workflow.add_conditional_edges(
            "router",
            self._route_based_on_intent,
            {
                "validate_schematic_input": "validate_schematic_input",
                "validate_search_input": "validate_search_input",
                "validate_bom_input": "validate_bom_input",
                "validate_parts_input": "validate_parts_input",
                "validate_bom_list_input": "validate_bom_list_input",  # New
                "handle_error": "handle_error"
            }
        )

        # Validation edges (add new BOM list validation)
        workflow.add_conditional_edges(
            "validate_schematic_input",
            self._check_validation_result,
            {"valid": "analyze_schematic", "invalid": "handle_error"}
        )
        workflow.add_conditional_edges(
            "validate_search_input",
            self._check_validation_result,
            {"valid": "search_components", "invalid": "handle_error"}
        )
        workflow.add_conditional_edges(
            "validate_bom_input",
            self._check_validation_result,
            {"valid": "create_bom", "invalid": "handle_error"}
        )
        workflow.add_conditional_edges(
            "validate_parts_input",
            self._check_validation_result,
            {"valid": "add_parts", "invalid": "handle_error"}
        )
        workflow.add_conditional_edges(
            "validate_bom_list_input",
            self._check_validation_result,
            {"valid": "list_boms", "invalid": "handle_error"}
        )

        # Operation to approval/response edges
        workflow.add_conditional_edges(
            "analyze_schematic",
            self._needs_approval,
            {"approval": "human_approval", "continue": "format_response", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "search_components",
            self._needs_approval,
            {"approval": "human_approval", "continue": "format_response", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "create_bom",
            self._needs_approval,
            {"approval": "human_approval", "continue": "format_response", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "add_parts",
            self._needs_approval,
            {"approval": "human_approval", "continue": "format_response", "error": "handle_error"}
        )
        workflow.add_conditional_edges(
            "list_boms",
            self._needs_approval,
            {"approval": "human_approval", "continue": "format_response", "error": "handle_error"}
        )

        # Approval handling
        workflow.add_conditional_edges(
            "human_approval",
            self._handle_approval_result,
            {
                "approved": "format_response",
                "rejected": "handle_error",
                "retry": "handle_retry",
                "timeout": "handle_error"
            }
        )

        # Error handling
        workflow.add_conditional_edges(
            "handle_error",
            self._should_retry,
            {"retry": "handle_retry", "final": "format_response"}
        )

        workflow.add_edge("handle_retry", "router")
        workflow.add_edge("format_response", END)

        workflow.set_entry_point("router")

        return workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["human_approval"]
        )

    async def _route_request(self, state: AgentState) -> AgentState:
        """Enhanced request routing with proper state access and null safety."""
        # Fix: Access state as dictionary, not object attributes
        messages = state.get("messages", [])
        user_input = ""

        if messages:
            last_message = messages[-1]
            user_input = getattr(last_message, 'content', '') or ""

        # Ensure user_input is a string
        if not isinstance(user_input, str):
            user_input = str(user_input) if user_input is not None else ""

        try:
            intent_result = await self._classify_intent_with_params(user_input)

            # Update state dictionary properly
            state["next_step"] = intent_result.intent
            state["extracted_params"] = intent_result.parameters
            state["confidence"] = intent_result.confidence
            state["validation_result"] = intent_result

            if not intent_result.is_valid:
                state["error"] = intent_result.error_message or "Intent classification failed"

        except Exception as e:
            error_msg = f"Intent classification failed: {str(e)}"
            state["error"] = error_msg
            state["next_step"] = "error"
            print(f"Route request error: {error_msg}")

        return state

    def _route_based_on_intent(self, state: AgentState) -> str:
        """Route based on classified intent with proper state access and null safety."""
        if state.get("error"):
            return "handle_error"

        intent = state.get("next_step", "").lower() if state.get("next_step") else "unknown"

        if intent in ["analyze_schematic", "analyze", "schematic"]:
            return "validate_schematic_input"
        elif intent in ["search_components", "search", "find"]:
            return "validate_search_input"
        elif intent in ["create_bom", "new_bom", "bom"]:
            return "validate_bom_input"
        elif intent in ["add_parts", "add", "parts"]:
            return "validate_parts_input"
        elif intent in ["show_boms", "list_boms", "get_boms", "show", "list"]:
            return "validate_bom_list_input"
        else:
            return "handle_error"

    def _check_validation_result(self, state: AgentState) -> str:
        """Check validation result with null safety."""
        validation_status = state.get("validation_status")
        return "valid" if validation_status == "valid" else "invalid"

    def _needs_approval(self, state: AgentState) -> str:
        """Check if step needs human approval with proper error handling."""
        if state.get("error"):
            return "error"
        elif state.get("requires_approval"):
            return "approval"
        else:
            return "continue"

    def _handle_approval_result(self, state: AgentState) -> str:
        """Handle human approval response with null safety."""
        approval_status = state.get("human_approval")

        if approval_status == "approved":
            return "approved"
        elif approval_status == "rejected":
            return "rejected"
        elif approval_status == "retry":
            return "retry"
        else:
            return "timeout"

    def _should_retry(self, state: AgentState) -> str:
        """Determine if operation should be retried with null safety."""
        return "retry" if state.get("should_retry") else "final"

    async def _validate_schematic_input(self, state: AgentState) -> AgentState:
        """Validate schematic analysis input with null safety."""
        params = state.get("extracted_params") or {}

        image_url = params.get("image_url")
        if not image_url:
            # Try to extract from user message
            user_input = ""
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                user_input = getattr(last_message, 'content', '') or ""

            if "http" in user_input:
                # Basic URL extraction
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                  user_input)
                if urls:
                    image_url = urls[0]

        if image_url:
            state["current_request"] = {"image_url": image_url}
            state["validation_status"] = "valid"
        else:
            state["validation_status"] = "invalid"
            state["error"] = "No image URL found for schematic analysis. Please provide a schematic image URL."

        return state

    async def _validate_search_input(self, state: AgentState) -> AgentState:
        """Validate component search input with null safety."""
        params = state.get("extracted_params") or {}

        if params.get("components") or params.get("search_terms"):
            state["current_request"] = params
            state["validation_status"] = "valid"
        else:
            # Try to use the full user input as search terms
            messages = state.get("messages", [])
            user_input = ""
            if messages:
                last_message = messages[-1]
                user_input = getattr(last_message, 'content', '') or ""

            if user_input:
                state["current_request"] = {"search_terms": user_input}
                state["validation_status"] = "valid"
            else:
                state["validation_status"] = "invalid"
                state["error"] = "No component search terms provided. Please specify what components to search for."

        return state

    async def _validate_bom_input(self, state: AgentState) -> AgentState:
        """Validate BOM creation input with null safety."""
        params = state.get("extracted_params") or {}

        bom_name = params.get("bom_name") or params.get("name")
        if bom_name and bom_name.strip():
            state["current_request"] = {
                "name": bom_name.strip(),
                "project": params.get("project", "").strip(),
                "description": params.get("description", "").strip()
            }
            state["validation_status"] = "valid"
        else:
            state["validation_status"] = "invalid"
            state["error"] = "BOM name is required for creation. Please specify a name for your BOM."

        return state

    async def _validate_parts_input(self, state: AgentState) -> AgentState:
        """Validate parts addition input with null safety."""
        params = state.get("extracted_params") or {}

        bom_name = params.get("bom_name")
        if bom_name and bom_name.strip():
            state["current_request"] = {
                "bom_name": bom_name.strip(),
                "project": params.get("project", "").strip(),
                "parts": params.get("parts", [])
            }
            state["validation_status"] = "valid"
        else:
            state["validation_status"] = "invalid"
            state["error"] = "BOM name is required for adding parts. Please specify which BOM to add parts to."

        return state

    async def _validate_bom_list_input(self, state: AgentState) -> AgentState:
        """Validate BOM listing input."""
        # For listing BOMs, we don't need specific parameters
        state["current_request"] = {"action": "list_boms"}
        state["validation_status"] = "valid"
        return state

    async def _list_boms(self, state: AgentState) -> AgentState:
        """List existing BOMs with comprehensive error handling."""
        try:
            # Ensure container is initialized
            if not hasattr(self, 'container') or not self.container._initialized:
                await self.container.initialize()

            result = await self.orchestrator.get_boms()

            if result and result.get("success"):
                state["step_result"] = result
                state["requires_approval"] = False  # No approval needed for listing
                # Clear any previous errors
                state["error"] = None
            else:
                error_msg = result.get("error", "Failed to retrieve BOMs") if result else "No response from BOM service"
                state["error"] = error_msg

        except Exception as e:
            error_msg = f"BOM listing error: {str(e)}"
            state["error"] = error_msg
            print(f"List BOMs error: {error_msg}")

        return state

    async def _analyze_schematic(self, state: AgentState) -> AgentState:
        """Analyze schematic with comprehensive error handling."""
        try:
            current_request = state.get("current_request") or {}
            image_url = current_request.get("image_url")

            if not image_url:
                state["error"] = "No image URL provided for schematic analysis"
                return state

            # Ensure container is initialized
            if not hasattr(self, 'container') or not self.container._initialized:
                await self.container.initialize()

            result = await self.orchestrator.analyze_schematic(image_url)

            if result and result.get("success"):
                state["step_result"] = result
                state["requires_approval"] = self.approval_config.require_analysis_approval

                if state["requires_approval"]:
                    approval_request = HumanApprovalRequest(
                        step="analyze_schematic",
                        message=f"Found {len(result.get('components', []))} components. Proceed with analysis?",
                        data=result,
                        auto_approve=self._should_auto_approve(result)
                    )
                    state["pending_approval"] = approval_request
                    state["requires_approval"] = not approval_request.auto_approve

                # Clear any previous errors
                state["error"] = None
            else:
                error_msg = result.get("error",
                                       "Schematic analysis failed") if result else "No response from schematic service"
                state["error"] = error_msg

        except Exception as e:
            error_msg = f"Schematic analysis error: {str(e)}"
            state["error"] = error_msg
            print(f"Analyze schematic error: {error_msg}")

        return state

    async def _search_components(self, state: AgentState) -> AgentState:
        """Search components with comprehensive error handling."""
        try:
            current_request = state.get("current_request") or {}
            components = current_request.get("components", [])
            search_terms = current_request.get("search_terms", "")

            if not components and not search_terms:
                state["error"] = "No components or search terms provided"
                return state

            # Ensure container is initialized
            if not hasattr(self, 'container') or not self.container._initialized:
                await self.container.initialize()

            # Convert search terms to components format if needed
            if search_terms and not components:
                components = [{"name": search_terms, "description": search_terms}]

            result = await self.orchestrator.search_components(components)

            if result and result.get("success"):
                state["step_result"] = result
                state["requires_approval"] = False  # Component search doesn't need approval
                # Clear any previous errors
                state["error"] = None
            else:
                error_msg = result.get("error",
                                       "Component search failed") if result else "No response from component service"
                state["error"] = error_msg

        except Exception as e:
            error_msg = f"Component search error: {str(e)}"
            state["error"] = error_msg
            print(f"Search components error: {error_msg}")

        return state

    async def _create_bom(self, state: AgentState) -> AgentState:
        """Create BOM with comprehensive error handling."""
        try:
            bom_data = state.get("current_request") or {}
            bom_name = bom_data.get("name", "").strip()

            if not bom_name:
                state["error"] = "BOM name is required for creation"
                return state

            # Ensure container is initialized
            if not hasattr(self, 'container') or not self.container._initialized:
                await self.container.initialize()

            result = await self.orchestrator.create_bom(bom_data)

            if result and result.get("success"):
                state["step_result"] = result
                state["requires_approval"] = self.approval_config.require_bom_creation_approval

                if state["requires_approval"]:
                    approval_request = HumanApprovalRequest(
                        step="create_bom",
                        message=f"Create BOM '{bom_name}'?",
                        data=result,
                        auto_approve=False
                    )
                    state["pending_approval"] = approval_request

                # Clear any previous errors
                state["error"] = None
            else:
                error_msg = result.get("error", "BOM creation failed") if result else "No response from BOM service"
                state["error"] = error_msg

        except Exception as e:
            error_msg = f"BOM creation error: {str(e)}"
            state["error"] = error_msg
            print(f"Create BOM error: {error_msg}")

        return state

    async def _add_parts(self, state: AgentState) -> AgentState:
        """Add parts with comprehensive error handling."""
        try:
            parts_data = state.get("current_request") or {}
            bom_name = parts_data.get("bom_name", "").strip()

            if not bom_name:
                state["error"] = "BOM name is required for adding parts"
                return state

            # Ensure container is initialized
            if not hasattr(self, 'container') or not self.container._initialized:
                await self.container.initialize()

            result = await self.orchestrator.add_parts_to_bom(parts_data)

            if result and result.get("success"):
                state["step_result"] = result
                state["requires_approval"] = self.approval_config.require_parts_addition_approval

                if state["requires_approval"]:
                    parts_count = len(parts_data.get('parts', []))
                    approval_request = HumanApprovalRequest(
                        step="add_parts",
                        message=f"Add {parts_count} parts to BOM '{bom_name}'?",
                        data=result,
                        auto_approve=self._should_auto_approve_parts(parts_data)
                    )
                    state["pending_approval"] = approval_request
                    state["requires_approval"] = not approval_request.auto_approve

                # Clear any previous errors
                state["error"] = None
            else:
                error_msg = result.get("error", "Adding parts failed") if result else "No response from BOM service"
                state["error"] = error_msg

        except Exception as e:
            error_msg = f"Add parts error: {str(e)}"
            state["error"] = error_msg
            print(f"Add parts error: {error_msg}")

        return state

    async def _handle_retry(self, state: AgentState) -> AgentState:
        """Handle retry logic with proper cleanup."""
        # Clear error state for retry
        state["error"] = None
        state["step_result"] = None
        state["requires_approval"] = False
        state["pending_approval"] = None

        # Reset validation status
        state["validation_status"] = None

        print(f"Retrying operation (attempt {state.get('retry_count', 0) + 1})")

        return state

    async def _classify_intent_with_params(self, user_input: str) -> IntentResult:
        """Enhanced intent classification with parameter extraction and robust error handling."""

        # First, try rule-based classification for simple cases to avoid LLM calls
        if self._is_simple_request(user_input):
            return self._fallback_intent_classification(user_input)

        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's request and extract their intent and parameters.

    IMPORTANT: Return ONLY valid JSON in this exact format:
    {
        "intent": "one_of_the_valid_intents",
        "parameters": {},
        "confidence": 0.9
    }

    Valid intents:
    - "analyze_schematic": For analyzing circuit schematics (look for image URLs, mentions of schematics)
    - "search_components": For searching electronic components (look for component names, part numbers)
    - "create_bom": For creating new BOMs (look for "create", "new BOM", BOM names)
    - "add_parts": For adding parts to existing BOMs (look for "add to", "add parts", existing BOM names)
    - "show_boms": For listing/showing existing BOMs (look for "show", "list", "get BOMs")
    - "unknown": For unclear requests

    For each intent, extract relevant parameters:
    - analyze_schematic: {"image_url": "url_if_found"}
    - search_components: {"components": ["component1", "component2"], "search_terms": "search query"}
    - create_bom: {"bom_name": "name", "project": "project", "description": "desc"}
    - add_parts: {"bom_name": "name", "project": "project"}
    - show_boms: {} (no parameters needed)

    Examples:
    User: "Analyze this schematic: https://example.com/image.jpg"
    Response: {"intent": "analyze_schematic", "parameters": {"image_url": "https://example.com/image.jpg"}, "confidence": 0.95}

    User: "Create a new BOM called PowerSupply for Arduino project"
    Response: {"intent": "create_bom", "parameters": {"bom_name": "PowerSupply", "project": "Arduino"}, "confidence": 0.9}

    User: "Show me all my BOMs"
    Response: {"intent": "show_boms", "parameters": {}, "confidence": 0.95}
    """),
            ("user", "{input}")
        ])

        response = None  # Initialize response variable
        try:
            # Ensure container is initialized
            if not self.container._initialized:
                await self.container.initialize()

            llm = self.container.get_llm()
            response = await llm.ainvoke(
                classification_prompt.format_messages(input=user_input)
            )

            content = response.content.strip()

            # More robust JSON extraction
            import json
            import re

            # Try to find JSON in the response
            json_patterns = [
                r'\{[^}]*"intent"[^}]*\}',  # Look for JSON with "intent" key
                r'\{.*?\}',  # Any JSON-like structure
            ]

            parsed_data = None
            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        parsed_data = json.loads(match)
                        if "intent" in parsed_data:  # Valid structure found
                            break
                    except json.JSONDecodeError:
                        continue
                if parsed_data and "intent" in parsed_data:
                    break

            if not parsed_data or "intent" not in parsed_data:
                # Fallback: Try to parse the entire content
                try:
                    parsed_data = json.loads(content)
                except json.JSONDecodeError:
                    # Last resort: Rule-based classification
                    return self._fallback_intent_classification(user_input)

            # Validate and clean the parsed data
            intent = parsed_data.get("intent", "unknown").lower().strip()
            parameters = parsed_data.get("parameters", {})
            confidence = float(parsed_data.get("confidence", 0.5))

            # Map alternative intent names to standard ones
            intent_mapping = {
                "analyze": "analyze_schematic",
                "schematic": "analyze_schematic",
                "search": "search_components",
                "find": "search_components",
                "components": "search_components",
                "create": "create_bom",
                "new_bom": "create_bom",
                "bom": "create_bom",
                "add": "add_parts",
                "parts": "add_parts",
                "show": "show_boms",
                "list": "show_boms",
                "get_boms": "show_boms",
                "list_boms": "show_boms"
            }

            intent = intent_mapping.get(intent, intent)

            # Validate intent
            valid_intents = ["analyze_schematic", "search_components", "create_bom", "add_parts", "show_boms",
                             "unknown"]
            if intent not in valid_intents:
                intent = "unknown"
                confidence = 0.3

            return IntentResult(
                intent=intent,
                confidence=min(1.0, max(0.0, confidence)),  # Clamp between 0 and 1
                parameters=parameters if isinstance(parameters, dict) else {},
                is_valid=True
            )

        except Exception as e:
            print(f"Intent classification error: {str(e)}")
            if response:
                print(f"LLM response content: {getattr(response, 'content', 'No response')[:200]}...")
            else:
                print("No LLM response received - likely container initialization issue")

            # Fallback to rule-based classification
            return self._fallback_intent_classification(user_input)

    def _is_simple_request(self, user_input: str) -> bool:
        """Check if this is a simple request that can be handled without LLM."""
        user_input_lower = user_input.lower().strip()

        simple_patterns = [
            "show me all my boms",
            "list boms",
            "get boms",
            "show boms",
            "list all boms",
            "display boms"
        ]

        return any(pattern in user_input_lower for pattern in simple_patterns)

    def _fallback_intent_classification(self, user_input: str) -> IntentResult:
        """Fallback rule-based intent classification when LLM fails."""
        user_input_lower = user_input.lower().strip()

        # Rule-based classification patterns
        if any(keyword in user_input_lower for keyword in ["schematic", "analyze", "image", "circuit"]):
            # Look for URLs in the input
            import re
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                              user_input)
            parameters = {"image_url": urls[0]} if urls else {}
            return IntentResult(
                intent="analyze_schematic",
                confidence=0.8,
                parameters=parameters,
                is_valid=True
            )

        elif any(keyword in user_input_lower for keyword in ["search", "find", "component", "part"]):
            return IntentResult(
                intent="search_components",
                confidence=0.8,
                parameters={"search_terms": user_input},
                is_valid=True
            )

        elif any(keyword in user_input_lower for keyword in ["create", "new bom", "make bom"]):
            # Try to extract BOM name
            words = user_input.split()
            bom_name = ""
            if "called" in user_input_lower:
                try:
                    called_idx = [i for i, w in enumerate(words) if w.lower() == "called"][0]
                    if called_idx + 1 < len(words):
                        bom_name = words[called_idx + 1]
                except (IndexError, ValueError):
                    pass

            return IntentResult(
                intent="create_bom",
                confidence=0.8,
                parameters={"bom_name": bom_name, "project": "", "description": ""},
                is_valid=True
            )

        elif any(keyword in user_input_lower for keyword in ["add parts", "add to", "add component"]):
            return IntentResult(
                intent="add_parts",
                confidence=0.8,
                parameters={"bom_name": "", "project": ""},
                is_valid=True
            )

        elif any(keyword in user_input_lower for keyword in
                 ["show", "list", "get", "display"]) and "bom" in user_input_lower:
            return IntentResult(
                intent="show_boms",
                confidence=0.9,  # High confidence for this simple pattern
                parameters={},
                is_valid=True
            )

        else:
            return IntentResult(
                intent="unknown",
                confidence=0.3,
                parameters={},
                is_valid=False,
                error_message=f"Could not understand the request: '{user_input}'. Please try rephrasing or use commands like 'show boms', 'create bom', 'search components', or 'analyze schematic'."
            )

    async def _request_human_approval(self, state: AgentState) -> AgentState:
        """Request human approval with timeout handling."""
        approval_request = state.get("pending_approval")
        if approval_request:
            state["awaiting_human_input"] = True
            # This will interrupt the graph execution

        return state

    async def _handle_error(self, state: AgentState) -> AgentState:
        """Centralized error handling with proper null checks."""
        error = state.get("error") or "Unknown error occurred"

        # Ensure error is a string
        if not isinstance(error, str):
            error = str(error) if error is not None else "Unknown error occurred"

        # Log error (in production, use proper logging)
        print(f"Agent Error: {error}")

        # Determine if error is retryable - with null safety
        retryable_errors = [
            "network", "timeout", "rate limit", "temporary", "authentication"
        ]

        error_lower = error.lower() if error else ""
        is_retryable = any(keyword in error_lower for keyword in retryable_errors)
        retry_count = state.get("retry_count", 0)

        if is_retryable and retry_count < 3:
            state["should_retry"] = True
            state["retry_count"] = retry_count + 1
        else:
            state["should_retry"] = False

        # Prepare error response
        state["step_result"] = {
            "success": False,
            "error": error,
            "retry_count": retry_count,
            "is_retryable": is_retryable
        }

        return state

    async def _format_response(self, state: AgentState) -> AgentState:
        """Format the final response with proper UI recommendations."""
        step_result = state.get("step_result", {})

        # Fix: Generate UI recommendations as UIRecommendation object, not dict
        ui_recommendations = self._generate_ui_recommendations(step_result)

        response = AgentResponse(
            success=not bool(state.get("error")),
            data=step_result,
            response_type=self._determine_response_type(step_result),
            error=state.get("error"),
            metadata={
                "session_id": self.container.session_id,
                "step": state.get("next_step"),
                "retry_count": state.get("retry_count", 0)
            },
            ui_recommendations=ui_recommendations  # This should be UIRecommendation object or None
        )

        state["final_response"] = response.to_dict()
        return state

    def _generate_ui_recommendations(self, result: Dict[str, Any]) -> Optional['UIRecommendation']:
        """Generate UI rendering recommendations as UIRecommendation object."""
        from src.models.state import UIRecommendation  # Import here to avoid circular imports

        if not result or not result.get("success", True):
            # For error cases
            return UIRecommendation(
                display_type=ResponseType.ERROR,
                actions=["retry", "modify_request"]
            )

        # Determine display type
        response_type = self._determine_response_type(result)

        if response_type == ResponseType.TABLE:
            # For table displays (components, search results)
            columns = self._extract_table_columns(result)
            return UIRecommendation(
                display_type=ResponseType.TABLE,
                columns=columns,
                sortable_columns=columns,
                filterable_columns=columns[:5] if len(columns) > 5 else columns,
                actions=["create_bom", "search_more_details", "export"]
            )

        elif response_type == ResponseType.TREE:
            # For tree displays (BOMs, projects)
            return UIRecommendation(
                display_type=ResponseType.TREE,
                actions=["expand_all", "collapse_all", "filter", "export"],
                grouping_options=["project", "manufacturer", "category"]
            )

        elif response_type == ResponseType.STATUS:
            # For status displays
            return UIRecommendation(
                display_type=ResponseType.STATUS,
                actions=["refresh", "clear"]
            )

        else:
            # Generic display
            return UIRecommendation(
                display_type=ResponseType.GENERIC,
                actions=self._suggest_next_actions(result)
            )

    def _determine_response_type(self, result: Dict[str, Any]) -> ResponseType:
        """Determine the appropriate response type for UI rendering."""
        if result.get("success") is False:
            return ResponseType.ERROR
        elif "components" in result:
            return ResponseType.TABLE
        elif "projects" in result or "boms" in result:
            return ResponseType.TREE
        else:
            return ResponseType.GENERIC

    def _extract_table_columns(self, result: Dict[str, Any]) -> List[str]:
        """Extract appropriate columns for table display."""
        if "components" in result and result["components"]:
            first_component = result["components"][0]
            return list(first_component.keys()) if isinstance(first_component, dict) else []
        return []

    def _suggest_next_actions(self, result: Dict[str, Any]) -> List[str]:
        """Suggest next possible actions based on result."""
        actions = []

        if result.get("success") is False:
            actions.extend(["retry", "modify_request"])
        elif "components" in result:
            actions.extend(["create_bom", "search_more_details"])
        elif "boms" in result:
            actions.extend(["add_parts", "export_bom"])

        return actions

    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate a human-readable summary."""
        if result.get("success") is False:
            return f"Error: {result.get('error', 'Operation failed')}"
        elif "components" in result:
            count = len(result["components"])
            return f"Found {count} components ready for BOM creation"
        elif "boms" in result:
            count = len(result.get("boms", []))
            return f"Retrieved {count} existing BOMs"
        else:
            return "Operation completed successfully"

    def _should_auto_approve(self, result: Dict[str, Any]) -> bool:
        """Determine if result should be auto-approved."""
        confidence = result.get("confidence", 0.0)
        return confidence > (self.approval_config.auto_approve_threshold or 0.95)

    def _should_auto_approve_parts(self, parts_data: Dict[str, Any]) -> bool:
        """Determine if parts addition should be auto-approved."""
        part_count = len(parts_data.get("parts", []))
        confidence = parts_data.get("confidence", 0.0)
        return part_count <= 5 and confidence > 0.9

    async def process_request(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process user request through the corrected graph."""
        initial_state = AgentState(
            messages=[HumanMessage(content=user_input)],
            session_id=session_id or self.container.session_id,
            current_request={},
            step_result=None,
            next_step=None,
            requires_approval=False,
            human_approval=None,
            human_feedback=None,
            pending_approval=None,
            awaiting_human_input=False,
            error=None,
            confidence=0.0,
            metadata={},
            final_response=None
        )

        config = {"configurable": {"thread_id": session_id or self.container.session_id}}

        try:
            result = await self.graph.ainvoke(initial_state, config)
            return result.get("final_response", {})
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_type": "error",
                "data": None
            }

    async def handle_human_approval(self, session_id: str, approval: bool,
                                    feedback: Optional[str] = None) -> Dict[str, Any]:
        """Handle human approval response."""
        config = {"configurable": {"thread_id": session_id}}

        approval_state = {
            "human_approval": "approved" if approval else "rejected",
            "human_feedback": feedback
        }

        try:
            result = await self.graph.ainvoke(approval_state, config)
            return result.get("final_response", {})
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_type": "error"
            }


# Factory function
async def create_bom_agent(config: AppConfig, session_id: Optional[str] = None) -> ModernBOMAgent:
    """Factory function to create and initialize the BOM agent."""
    agent = ModernBOMAgent(config, session_id)
    return agent
