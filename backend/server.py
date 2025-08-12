# server.py - FastAPI server with human-in-the-loop support
"""FastAPI server for BOM agent with real-time human approval."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

from src.core.config import AppConfig
from main import ModernBOMAgent
from src.models.state import HumanApprovalRequest


class ChatRequest(BaseModel):
    """Enhanced chat request with validation."""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 10000:
            raise ValueError('Message too long (max 10000 characters)')
        return v.strip()


class ApprovalResponse(BaseModel):
    """Enhanced approval response with validation."""
    approval_id: str
    approved: bool
    feedback: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None

    @validator('feedback')
    def validate_feedback(cls, v):
        if v and len(v) > 1000:
            raise ValueError('Feedback too long (max 1000 characters)')
        return v


class ChatResponse(BaseModel):
    """Enhanced chat response."""
    id: str
    success: bool
    response_type: str
    data: Optional[Any] = None
    message: Optional[str] = None
    ui_recommendations: Optional[Dict[str, Any]] = None
    approval_request: Optional[Dict[str, Any]] = None
    session_id: str
    timestamp: str
    error_code: Optional[str] = None


class SessionManager:
    """Thread-safe session management with proper async initialization."""

    def __init__(self):
        self._agents: Dict[str, ModernBOMAgent] = {}
        self._pending_approvals: Dict[str, HumanApprovalRequest] = {}
        self._session_activity: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._initialized = False

    async def initialize(self):
        """Initialize the session manager with event loop."""
        if not self._initialized:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self._initialized = True

    async def _periodic_cleanup(self):
        """Clean up inactive sessions."""
        while True:
            try:
                await asyncio.sleep(1800)  # Check every 30 minutes
                await self._cleanup_inactive_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Session cleanup error: {e}")

    async def _cleanup_inactive_sessions(self):
        """Remove inactive sessions (older than 2 hours)."""
        async with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=2)
            inactive_sessions = [
                session_id for session_id, last_activity in self._session_activity.items()
                if last_activity < cutoff_time
            ]

            for session_id in inactive_sessions:
                if session_id in self._agents:
                    await self._agents[session_id].container.cleanup()
                    del self._agents[session_id]

                self._session_activity.pop(session_id, None)

                # Clean up related approvals
                approvals_to_remove = [
                    approval_id for approval_id, approval in self._pending_approvals.items()
                    if approval.metadata.get("session_id") == session_id
                ]
                for approval_id in approvals_to_remove:
                    del self._pending_approvals[approval_id]

    async def get_or_create_agent(self, session_id: str) -> ModernBOMAgent:
        """Thread-safe agent creation."""
        async with self._lock:
            if session_id not in self._agents:
                config = AppConfig.from_env()
                agent = ModernBOMAgent(config, session_id)
                await agent.container.initialize()
                self._agents[session_id] = agent

            self._session_activity[session_id] = datetime.now()
            return self._agents[session_id]

    async def add_pending_approval(self, approval_req: HumanApprovalRequest):
        """Add pending approval."""
        async with self._lock:
            self._pending_approvals[approval_req.id] = approval_req

    async def get_pending_approval(self, approval_id: str) -> Optional[HumanApprovalRequest]:
        """Get and remove pending approval."""
        async with self._lock:
            return self._pending_approvals.pop(approval_id, None)

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        async with self._lock:
            return {
                "active_sessions": len(self._agents),
                "pending_approvals": len(self._pending_approvals),
                "sessions": list(self._agents.keys())
            }

    async def shutdown(self):
        """Shutdown the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cleanup all sessions
        async with self._lock:
            for agent in self._agents.values():
                try:
                    await agent.container.cleanup()
                except Exception as e:
                    print(f"Cleanup error: {e}")

        self._initialized = False


# Global session manager - will be initialized in lifespan
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan manager."""
    print("🚀 Enhanced BOM Agent API starting up...")

    # Initialize session manager
    await session_manager.initialize()

    # Validate configuration
    config = AppConfig.from_env()
    validation_errors = config.validate()
    if validation_errors:
        print("❌ Configuration errors:")
        for error in validation_errors:
            print(f"  - {error}")
        raise RuntimeError("Invalid configuration")

    print("✅ Configuration validated")
    print("✅ Session manager initialized")

    yield

    print("🔄 BOM Agent API shutting down...")

    # Shutdown session manager
    await session_manager.shutdown()

    print("✅ Cleanup completed")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced BOM Agent API",
    description="Intelligent BOM Management Agent with Human-in-the-Loop",
    version="2.1.0",
    lifespan=lifespan
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header."""
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.post("/api/chat", response_model=ChatResponse)
async def enhanced_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Enhanced chat endpoint with proper validation and error handling."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        agent = await session_manager.get_or_create_agent(session_id)

        # Process the request
        result = await agent.process_request(request.message, session_id)

        # Handle approval requests
        if result.get("approval_request"):
            approval_req = HumanApprovalRequest(**result["approval_request"])
            approval_req.metadata["session_id"] = session_id
            await session_manager.add_pending_approval(approval_req)

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
            timestamp=datetime.now().isoformat(),
            error_code=result.get("error_code")
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/approval/{approval_id}", response_model=ChatResponse)
async def handle_enhanced_approval(approval_id: str, response: ApprovalResponse):
    """Enhanced approval handling with validation."""
    try:
        approval_request = await session_manager.get_pending_approval(approval_id)
        if not approval_request:
            raise HTTPException(status_code=404, detail="Approval request not found or expired")

        # Get the agent for this session
        session_id = approval_request.metadata.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Invalid approval request")

        agent = await session_manager.get_or_create_agent(session_id)

        # Continue the workflow with human approval
        result = await agent.handle_human_approval(
            session_id,
            response.approved,
            response.feedback
        )

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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Approval processing error: {str(e)}")


@app.get("/api/sessions")
async def list_sessions():
    """List active sessions with statistics."""
    try:
        stats = await session_manager.get_session_stats()
        return {
            "success": True,
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
async def terminate_session(session_id: str):
    """Terminate a specific session."""
    try:
        async with session_manager._lock:
            if session_id in session_manager._agents:
                agent = session_manager._agents[session_id]
                await agent.container.cleanup()
                del session_manager._agents[session_id]
                session_manager._session_activity.pop(session_id, None)

                # Clean up related approvals
                approvals_to_remove = [
                    approval_id for approval_id, approval in session_manager._pending_approvals.items()
                    if approval.metadata.get("session_id") == session_id
                ]
                for approval_id in approvals_to_remove:
                    del session_manager._pending_approvals[approval_id]

                return {"success": True, "message": f"Session {session_id} terminated"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def enhanced_health_check():
    """Enhanced health check with system status."""
    try:
        stats = await session_manager.get_session_stats()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "active_sessions": stats["active_sessions"],
                "pending_approvals": stats["pending_approvals"]
            },
            "version": "2.1.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)