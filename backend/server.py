# ================================================================================================
# FASTAPI SERVER - Clean & Simple
# ================================================================================================
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

from config import AppConfig
from main import SimpleBOMAgent


# Request/Response models for API
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 5000:  # Reasonable limit
            raise ValueError('Message too long (max 5000 characters)')
        return v.strip()


class ChatResponse(BaseModel):
    id: str
    success: bool
    type: str
    message: str
    data: Optional[Dict[str, Any]] = None
    suggestions: List[str] = []
    timestamp: str


class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]
    total_count: int


class SessionInfoResponse(BaseModel):
    session_id: str
    history_count: int
    initialized: bool
    last_activity: Optional[str]


# Enhanced session management
class SessionManager:
    def __init__(self):
        self._agents: Dict[str, SimpleBOMAgent] = {}
        self._last_activity: Dict[str, datetime] = {}

    async def get_agent(self, session_id: str, config) -> SimpleBOMAgent:
        """Get or create agent for session"""
        if session_id not in self._agents:
            agent = SimpleBOMAgent(config, session_id)
            await agent.initialize()
            self._agents[session_id] = agent

        # Update last activity
        self._last_activity[session_id] = datetime.now()
        return self._agents[session_id]

    async def cleanup_session(self, session_id: str) -> bool:
        """Cleanup specific session"""
        if session_id in self._agents:
            await self._agents[session_id].cleanup()
            del self._agents[session_id]
            self._last_activity.pop(session_id, None)
            return True
        return False

    async def cleanup_inactive_sessions(self, max_age_hours: int = 2):
        """Cleanup sessions older than max_age_hours"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        inactive_sessions = [
            sid for sid, last_time in self._last_activity.items()
            if last_time < cutoff
        ]

        for session_id in inactive_sessions:
            await self.cleanup_session(session_id)

        return len(inactive_sessions)

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self._agents.keys())

    async def cleanup_all(self):
        """Cleanup all sessions"""
        for agent in self._agents.values():
            await agent.cleanup()
        self._agents.clear()
        self._last_activity.clear()


# Create FastAPI app
app = FastAPI(
    title="BOM Agent API",
    description="Simplified BOM Management Agent with React Pattern",
    version="2.0.0"
)

# CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Global session manager
session_manager = SessionManager()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup"""
    try:
        config = AppConfig.from_env()
        validation_errors = config.validate()

        if validation_errors:
            print("❌ Configuration errors:")
            for error in validation_errors:
                print(f"  - {error}")
            raise RuntimeError("Invalid configuration")

        print("✅ BOM Agent API started successfully")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await session_manager.cleanup_all()
    print("👋 BOM Agent API shut down")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint for Angular UI"""
    try:
        # Load configuration
        config = AppConfig.from_env()

        # Get or create agent for session
        session_id = request.session_id or str(uuid.uuid4())
        agent = await session_manager.get_agent(session_id, config)

        # Process request
        response = await agent.process_request(request.message)

        # Schedule cleanup of inactive sessions in background
        background_tasks.add_task(session_manager.cleanup_inactive_sessions)

        return ChatResponse(**response.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get conversation history for session"""
    try:
        if session_id in session_manager._agents:
            agent = session_manager._agents[session_id]
            history = await agent.get_conversation_history()
            return HistoryResponse(
                session_id=session_id,
                history=history,
                total_count=len(history)
            )
        else:
            return HistoryResponse(session_id=session_id, history=[], total_count=0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for session"""
    try:
        if session_id in session_manager._agents:
            agent = session_manager._agents[session_id]
            await agent.clear_history()
            return {"message": f"History cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """Get session information"""
    try:
        if session_id in session_manager._agents:
            agent = session_manager._agents[session_id]
            info = await agent.get_session_info()
            return SessionInfoResponse(**info)
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
async def cleanup_session(session_id: str):
    """Cleanup specific session"""
    try:
        success = await session_manager.cleanup_session(session_id)
        if success:
            return {"message": f"Session {session_id} cleaned up successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    try:
        active_sessions = session_manager.get_active_sessions()
        return {
            "active_sessions": active_sessions,
            "total_count": len(active_sessions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/cleanup")
async def cleanup_inactive_sessions(max_age_hours: int = 2):
    """Cleanup inactive sessions"""
    try:
        cleaned_count = await session_manager.cleanup_inactive_sessions(max_age_hours)
        return {
            "message": f"Cleaned up {cleaned_count} inactive sessions",
            "cleaned_sessions": cleaned_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        active_sessions = len(session_manager.get_active_sessions())

        return {
            "status": "healthy",
            "active_sessions": active_sessions,
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "session_manager": "operational",
                "agent_factory": "operational"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Run server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)