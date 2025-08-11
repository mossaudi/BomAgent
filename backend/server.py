# server.py - FastAPI server with human-in-the-loop support
"""FastAPI server for BOM agent with real-time human approval."""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import AppConfig
from src.main import ModernBOMAgent
from src.models.state import HumanApprovalRequest


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ApprovalResponse(BaseModel):
    """Human approval response model."""
    approval_id: str
    approved: bool
    feedback: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    id: str
    success: bool
    response_type: str
    data: Optional[Any] = None
    message: Optional[str] = None
    ui_recommendations: Optional[Dict[str, Any]] = None
    approval_request: Optional[Dict[str, Any]] = None
    session_id: str
    timestamp: str


# Global storage for pending approvals (In production, use Redis or database)
pending_approvals: Dict[str, HumanApprovalRequest] = {}
agent_instances: Dict[str, ModernBOMAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 BOM Agent API starting up...")
    yield
    # Shutdown
    print("🔄 BOM Agent API shutting down...")
    # Cleanup agent instances
    for agent in agent_instances.values():
        await agent.container.cleanup()


# Initialize FastAPI app
app = FastAPI(
    title="BOM Agent API",
    description="Intelligent BOM Management Agent with Human-in-the-Loop",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_or_create_agent(session_id: str) -> ModernBOMAgent:
    """Get or create agent instance for session."""
    if session_id not in agent_instances:
        config = AppConfig.from_env()
        agent = ModernBOMAgent(config, session_id)
        await agent.container.initialize()
        agent_instances[session_id] = agent

    return agent_instances[session_id]


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint with human-in-the-loop support."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        agent = await get_or_create_agent(session_id)

        # Process the request
        result = await agent.process_request(request.message, session_id)

        # Check if human approval is needed
        if result.get("approval_request"):
            approval_req = HumanApprovalRequest(**result["approval_request"])
            pending_approvals[approval_req.id] = approval_req

            return ChatResponse(
                id=str(uuid.uuid4()),
                success=True,
                response_type="approval_request",
                data=result.get("data"),
                message="Human approval required",
                approval_request=approval_req.to_dict(),
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )

        # Return normal response
        return ChatResponse(
            id=str(uuid.uuid4()),
            success=result.get("success", True),
            response_type=result.get("response_type", "generic"),
            data=result.get("data"),
            message=result.get("message"),
            ui_recommendations=result.get("ui_recommendations"),
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/approval/{approval_id}")
async def handle_approval(approval_id: str, response: ApprovalResponse):
    """Handle human approval response."""
    try:
        if approval_id not in pending_approvals:
            raise HTTPException(status_code=404, detail="Approval request not found")

        approval_request = pending_approvals[approval_id]

        # Get the agent for this session
        # Note: In production, you'd need to track session_id with approval_id
        session_id = approval_request.metadata.get("session_id", "default")
        agent = await get_or_create_agent(session_id)

        # Continue the workflow with human approval
        result = await agent.handle_human_approval(
            session_id,
            response.approved,
            response.feedback
        )

        # Remove from pending approvals
        del pending_approvals[approval_id]

        return ChatResponse(
            id=str(uuid.uuid4()),
            success=result.get("success", True),
            response_type=result.get("response_type", "generic"),
            data=result.get("data"),
            message=result.get("message", "Approval processed"),
            ui_recommendations=result.get("ui_recommendations"),
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/{session_id}")
async def get_memory_status(session_id: str):
    """Get memory status for a session."""
    try:
        agent = await get_or_create_agent(session_id)
        status = await agent.orchestrator.get_memory_status()

        return ChatResponse(
            id=str(uuid.uuid4()),
            success=True,
            response_type="status",
            data=status,
            message="Memory status retrieved",
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear memory for a session."""
    try:
        if session_id in agent_instances:
            agent = agent_instances[session_id]
            await agent.container.get_memory_service().cleanup()
            del agent_instances[session_id]

        return {"success": True, "message": f"Memory cleared for session {session_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    """List active sessions."""
    return {
        "sessions": list(agent_instances.keys()),
        "count": len(agent_instances),
        "pending_approvals": len(pending_approvals)
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(agent_instances),
        "pending_approvals": len(pending_approvals)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)