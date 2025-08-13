# src/models/models.py
"""Modern state management and response models for LangGraph agent."""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union, TypedDict, Annotated
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ResponseType(Enum):
    """Types of responses that determine UI rendering."""
    TABLE = "table"
    TREE = "tree"
    CHART = "chart"
    STATUS = "status"
    FORM = "form"
    GENERIC = "generic"
    ERROR = "error"


class InteractionType(Enum):
    """Types of user interactions."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    CANCEL = "cancel"


class AgentState(TypedDict):
    """Modern LangGraph state with proper typing."""
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    current_request: Dict[str, Any]
    step_result: Optional[Dict[str, Any]]
    next_step: Optional[str]
    requires_approval: bool
    human_approval: Optional[str]
    human_feedback: Optional[str]
    pending_approval: Optional[HumanApprovalRequest]
    awaiting_human_input: bool
    error: Optional[str]
    confidence: float
    metadata: Dict[str, Any]
    final_response: Optional[Dict[str, Any]]


@dataclass
class HumanApprovalRequest:
    """Request for human approval in the workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step: str = ""
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    auto_approve: bool = False
    timeout_seconds: int = 300  # 5 minutes default
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "step": self.step,
            "message": self.message,
            "data": self.data,
            "auto_approve": self.auto_approve,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class UserInteraction:
    """User interaction response to approval requests."""
    approval_id: str
    action: InteractionType
    approved: bool
    feedback: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "action": self.action.value,
            "approved": self.approved,
            "feedback": self.feedback,
            "modifications": self.modifications,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ComponentData:
    """Standardized component data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    value: Optional[str] = None
    quantity: int = 1
    designator: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "part_number": self.part_number,
            "manufacturer": self.manufacturer,
            "description": self.description,
            "value": self.value,
            "quantity": self.quantity,
            "designator": self.designator,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class BOMData:
    """BOM structure for UI consumption."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    project: Optional[str] = None
    components: List[ComponentData] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "project": self.project,
            "components": [comp.to_dict() for comp in self.components],
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "metadata": self.metadata,
            "component_count": len(self.components)
        }


@dataclass
class UIRecommendation:
    """UI rendering recommendations."""
    display_type: ResponseType
    columns: Optional[List[str]] = None
    sortable_columns: Optional[List[str]] = None
    filterable_columns: Optional[List[str]] = None
    grouping_options: Optional[List[str]] = None
    actions: List[str] = field(default_factory=list)
    export_formats: List[str] = field(default_factory=lambda: ["csv", "excel", "json"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_type": self.display_type.value,
            "columns": self.columns,
            "sortable_columns": self.sortable_columns,
            "filterable_columns": self.filterable_columns,
            "grouping_options": self.grouping_options,
            "actions": self.actions,
            "export_formats": self.export_formats
        }


@dataclass
class AgentResponse:
    """Standardized agent response for UI consumption."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = True
    response_type: ResponseType = ResponseType.GENERIC
    data: Optional[Union[Dict[str, Any], List[Any]]] = None
    error: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ui_recommendations: Optional[UIRecommendation] = None
    next_actions: List[str] = field(default_factory=list)
    approval_request: Optional[HumanApprovalRequest] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "success": self.success,
            "response_type": self.response_type.value,
            "data": self.data,
            "error": self.error,
            "message": self.message,
            "metadata": self.metadata,
            "ui_recommendations": self.ui_recommendations.to_dict() if self.ui_recommendations else None,
            "next_actions": self.next_actions,
            "approval_request": self.approval_request.to_dict() if self.approval_request else None,
            "timestamp": self.timestamp.isoformat()
        }


# Response builders for common use cases
class ResponseBuilder:
    """Builder pattern for creating standardized responses."""

    @staticmethod
    def table_response(
            data: List[Dict[str, Any]],
            columns: Optional[List[str]] = None,
            message: Optional[str] = None
    ) -> AgentResponse:
        """Build a table response."""
        if not columns and data:
            columns = list(data[0].keys()) if data else []

        ui_rec = UIRecommendation(
            display_type=ResponseType.TABLE,
            columns=columns,
            sortable_columns=columns,
            filterable_columns=columns[:5],  # Limit filterable columns
            actions=["export", "create_bom", "search_details"]
        )

        return AgentResponse(
            response_type=ResponseType.TABLE,
            data=data,
            message=message or f"Found {len(data)} items",
            ui_recommendations=ui_rec,
            next_actions=["create_bom", "add_to_existing_bom"]
        )

    @staticmethod
    def tree_response(
            data: Dict[str, Any],
            message: Optional[str] = None
    ) -> AgentResponse:
        """Build a tree response."""
        ui_rec = UIRecommendation(
            display_type=ResponseType.TREE,
            actions=["expand_all", "collapse_all", "filter", "export"]
        )

        return AgentResponse(
            response_type=ResponseType.TREE,
            data=data,
            message=message or "Hierarchical data ready",
            ui_recommendations=ui_rec,
            next_actions=["select_item", "create_new"]
        )

    @staticmethod
    def status_response(
            status: Dict[str, Any],
            message: Optional[str] = None
    ) -> AgentResponse:
        """Build a status response."""
        ui_rec = UIRecommendation(
            display_type=ResponseType.STATUS,
            actions=["refresh", "clear", "export"]
        )

        return AgentResponse(
            response_type=ResponseType.STATUS,
            data=status,
            message=message or "Status information",
            ui_recommendations=ui_rec
        )

    @staticmethod
    def error_response(
            error: str,
            details: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Build an error response."""
        return AgentResponse(
            success=False,
            response_type=ResponseType.ERROR,
            error=error,
            data=details,
            ui_recommendations=UIRecommendation(
                display_type=ResponseType.ERROR,
                actions=["retry", "cancel"]
            )
        )

    @staticmethod
    def approval_response(
            approval_request: HumanApprovalRequest,
            message: Optional[str] = None
    ) -> AgentResponse:
        """Build an approval request response."""
        return AgentResponse(
            response_type=ResponseType.FORM,
            message=message or "Approval required",
            approval_request=approval_request,
            ui_recommendations=UIRecommendation(
                display_type=ResponseType.FORM,
                actions=["approve", "reject", "modify"]
            )
        )

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
        result = asdict(self)
        result['type'] = self.type.value  # Convert enum to string
        return result