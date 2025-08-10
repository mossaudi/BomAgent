# main.py - MODERN LANGGRAPH AGENT
"""Modern LangGraph agent with clean architecture and human-in-the-loop patterns."""

import asyncio
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import AppConfig
from src.core.container import Container
from src.core.exceptions import AgentError
from src.models.state import AgentState, UserInteraction, ResponseType
from src.models.responses import AgentResponse, HumanApprovalRequest
from src.core.services.agent_service import AgentOrchestrator


class WorkflowStep(Enum):
    """Workflow step enumeration."""
    ANALYZE_SCHEMATIC = "analyze_schematic"
    SEARCH_COMPONENTS = "search_components"
    CREATE_BOM = "create_bom"
    ADD_PARTS = "add_parts"
    HUMAN_APPROVAL = "human_approval"


@dataclass
class HumanApprovalConfig:
    """Configuration for human approval requirements."""
    require_analysis_approval: bool = True
    require_bom_creation_approval: bool = True
    require_parts_addition_approval: bool = True
    auto_approve_threshold: Optional[float] = 0.95  # Auto-approve if confidence > 95%


class ModernBOMAgent:
    """Modern LangGraph agent with human-in-the-loop and clean architecture."""

    def __init__(self, config: AppConfig, session_id: Optional[str] = None):
        self.config = config
        self.container = Container(config, session_id)
        self.orchestrator = AgentOrchestrator(self.container)
        self.approval_config = HumanApprovalConfig()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with human-in-the-loop."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("router", self._route_request)
        workflow.add_node("analyze_schematic", self._analyze_schematic)
        workflow.add_node("search_components", self._search_components)
        workflow.add_node("create_bom", self._create_bom)
        workflow.add_node("add_parts", self._add_parts)
        workflow.add_node("human_approval", self._request_human_approval)
        workflow.add_node("format_response", self._format_response)

        # Add edges
        workflow.add_edge("router", "analyze_schematic")
        workflow.add_edge("analyze_schematic", "human_approval")
        workflow.add_edge("search_components", "human_approval")
        workflow.add_edge("create_bom", "human_approval")
        workflow.add_edge("add_parts", "human_approval")
        workflow.add_conditional_edges(
            "human_approval",
            self._should_continue_after_approval,
            {
                "approved": "format_response",
                "rejected": "router",
                "continue": "format_response"
            }
        )
        workflow.add_edge("format_response", END)

        # Set entry point
        workflow.set_entry_point("router")

        return workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["human_approval"]  # Always interrupt for human approval
        )

    async def _route_request(self, state: AgentState) -> Dict[str, Any]:
        """Route the request to appropriate handler."""
        user_input = state.messages[-1].content if state.messages else ""

        # Use LLM to classify intent
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify the user's intent. Return ONLY one of these actions:
            - analyze_schematic: If user wants to analyze a schematic image
            - search_components: If user wants to search for component data
            - create_bom: If user wants to create a new BOM
            - add_parts: If user wants to add parts to existing BOM
            - get_memory: If user wants to check memory/session status
            """),
            ("user", "{input}")
        ])

        llm = self.container.get_llm()
        response = await llm.ainvoke(
            classification_prompt.format_messages(input=user_input)
        )

        intent = response.content.strip().lower()

        return {
            "next_step": intent,
            "confidence": 0.8,  # Could be enhanced with confidence scoring
            "metadata": {"classification": intent}
        }

    async def _analyze_schematic(self, state: AgentState) -> Dict[str, Any]:
        """Analyze schematic with human approval."""
        try:
            result = await self.orchestrator.analyze_schematic(
                state.current_request.get("image_url", "")
            )

            approval_request = HumanApprovalRequest(
                step="analyze_schematic",
                message=f"Found {len(result.get('components', []))} components. Proceed?",
                data=result,
                auto_approve=self._should_auto_approve(result)
            )

            return {
                "pending_approval": approval_request,
                "step_result": result,
                "requires_approval": not approval_request.auto_approve
            }

        except Exception as e:
            return {
                "error": str(e),
                "step": "analyze_schematic",
                "requires_approval": False
            }

    async def _search_components(self, state: AgentState) -> Dict[str, Any]:
        """Search component data."""
        try:
            components = state.current_request.get("components", [])
            result = await self.orchestrator.search_components(components)

            return {
                "step_result": result,
                "requires_approval": False
            }

        except Exception as e:
            return {"error": str(e), "step": "search_components"}

    async def _create_bom(self, state: AgentState) -> Dict[str, Any]:
        """Create BOM with approval."""
        try:
            bom_data = state.current_request
            result = await self.orchestrator.create_bom(bom_data)

            approval_request = HumanApprovalRequest(
                step="create_bom",
                message=f"Create BOM '{bom_data.get('name')}'?",
                data=result,
                auto_approve=False  # Always require approval for BOM creation
            )

            return {
                "pending_approval": approval_request,
                "step_result": result,
                "requires_approval": True
            }

        except Exception as e:
            return {"error": str(e), "step": "create_bom"}

    async def _add_parts(self, state: AgentState) -> Dict[str, Any]:
        """Add parts to BOM with approval."""
        try:
            parts_data = state.current_request
            result = await self.orchestrator.add_parts_to_bom(parts_data)

            approval_request = HumanApprovalRequest(
                step="add_parts",
                message=f"Add {len(parts_data.get('parts', []))} parts to BOM?",
                data=result,
                auto_approve=self._should_auto_approve_parts(parts_data)
            )

            return {
                "pending_approval": approval_request,
                "step_result": result,
                "requires_approval": not approval_request.auto_approve
            }

        except Exception as e:
            return {"error": str(e), "step": "add_parts"}

    async def _request_human_approval(self, state: AgentState) -> Dict[str, Any]:
        """Request human approval for the current step."""
        if not state.get("requires_approval", False):
            return {"approval_status": "auto_approved"}

        # This will interrupt the graph execution
        return {
            "awaiting_human_input": True,
            "approval_request": state.get("pending_approval")
        }

    def _should_continue_after_approval(self, state: AgentState) -> Literal["approved", "rejected", "continue"]:
        """Determine next step after human approval."""
        if state.get("human_approval") == "approved":
            return "approved"
        elif state.get("human_approval") == "rejected":
            return "rejected"
        else:
            return "continue"

    async def _format_response(self, state: AgentState) -> Dict[str, Any]:
        """Format the final response as JSON for UI consumption."""
        step_result = state.get("step_result", {})

        response = AgentResponse(
            success=not bool(state.get("error")),
            data=step_result,
            response_type=self._determine_response_type(step_result),
            metadata={
                "session_id": self.container.session_id,
                "step": state.get("next_step"),
                "timestamp": state.get("timestamp")
            },
            ui_recommendations=self._generate_ui_recommendations(step_result)
        )

        return {"final_response": response.to_dict()}

    def _determine_response_type(self, result: Dict[str, Any]) -> ResponseType:
        """Determine the appropriate response type for UI rendering."""
        if "components" in result:
            return ResponseType.TABLE
        elif "projects" in result or "boms" in result:
            return ResponseType.TREE
        elif "status" in result:
            return ResponseType.STATUS
        else:
            return ResponseType.GENERIC

    def _generate_ui_recommendations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI rendering recommendations."""
        return {
            "display_type": self._determine_response_type(result).value,
            "columns": self._extract_table_columns(result) if "components" in result else None,
            "actions": self._suggest_next_actions(result),
            "summary": self._generate_summary(result)
        }

    def _extract_table_columns(self, result: Dict[str, Any]) -> List[str]:
        """Extract appropriate columns for table display."""
        if "components" in result and result["components"]:
            # Return column names based on first component
            first_component = result["components"][0]
            return list(first_component.keys()) if isinstance(first_component, dict) else []
        return []

    def _suggest_next_actions(self, result: Dict[str, Any]) -> List[str]:
        """Suggest next possible actions based on result."""
        actions = []

        if "components" in result:
            actions.extend(["create_bom", "search_more_details"])
        if "boms" in result:
            actions.extend(["add_parts", "export_bom"])

        return actions

    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate a human-readable summary."""
        if "components" in result:
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
        # Auto-approve if small number of parts and high confidence
        part_count = len(parts_data.get("parts", []))
        confidence = parts_data.get("confidence", 0.0)

        return part_count <= 5 and confidence > 0.9

    async def process_request(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process user request through the graph."""
        initial_state = AgentState(
            messages=[HumanMessage(content=user_input)],
            session_id=session_id or self.container.session_id,
            current_request={"input": user_input},
            metadata={}
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

        # Update state with human approval
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


# Factory function for easy instantiation
async def create_bom_agent(config: AppConfig, session_id: Optional[str] = None) -> ModernBOMAgent:
    """Factory function to create and initialize the BOM agent."""
    agent = ModernBOMAgent(config, session_id)
    return agent


# Main execution
async def main():
    """Main entry point for the agent."""
    config = AppConfig.from_env()
    agent = await create_bom_agent(config)

    # Example usage
    response = await agent.process_request("analyze schematic at http://example.com/schematic.png")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())