# server_corrected.py - Fixed Server with Proper Agent Interface
"""
Corrected server implementation that:
1. Uses the correct agent method names
2. Properly handles structured responses from agent
3. Returns JSON data for UI consumption (not formatted text)
4. Implements proper error handling and timeouts
"""

import asyncio
import signal
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any, Set

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import AppConfig
from main import BOMAgent, AgentResponse


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

    @field_validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()


class ChatResponse(BaseModel):
    """Response model matching the AgentResponse structure"""
    success: bool
    action: str
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]
    total_count: int


class SessionInfoResponse(BaseModel):
    session_id: str
    history_count: int
    components_count: int
    initialized: bool
    last_activity: Optional[str]


# Fixed session management
class SessionManager:
    """Clean session manager with proper agent interface"""

    def __init__(self):
        self._agents: Dict[str, BOMAgent] = {}
        self._agent_locks: Dict[str, asyncio.Lock] = {}
        self._active_requests: Set[asyncio.Task] = set()
        self._main_lock = asyncio.Lock()
        self._config: Optional[AppConfig] = None

    def set_config(self, config: AppConfig):
        """Set configuration after validation"""
        self._config = config

    async def get_agent(self, session_id: str) -> BOMAgent:
        """Get or create agent with lazy initialization"""
        if not self._config:
            raise HTTPException(status_code=500, detail="Configuration not loaded")

        # Get or create session-specific lock
        async with self._main_lock:
            if session_id not in self._agent_locks:
                self._agent_locks[session_id] = asyncio.Lock()
            session_lock = self._agent_locks[session_id]

        # Use session-specific lock for agent creation
        async with session_lock:
            if session_id not in self._agents:
                try:
                    print(f"🔄 Creating new agent for session {session_id}")
                    agent = BOMAgent(self._config, session_id)

                    # Initialize with timeout
                    await asyncio.wait_for(agent.initialize(), timeout=30.0)

                    self._agents[session_id] = agent
                    print(f"✅ Agent created successfully for session {session_id}")

                except asyncio.TimeoutError:
                    print(f"⌛ Agent initialization timeout for session {session_id}")
                    raise HTTPException(
                        status_code=504,
                        detail="Agent initialization timeout"
                    )
                except Exception as e:
                    print(f"❌ Failed to initialize agent for session {session_id}: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Agent initialization failed: {str(e)}"
                    )

            return self._agents[session_id]

    async def process_request_with_timeout(self, session_id: str, message: str) -> AgentResponse:
        """Process request with proper timeout and return structured response"""

        agent = await self.get_agent(session_id)

        # Create the request task
        request_task = asyncio.create_task(agent.process_request(message))
        self._active_requests.add(request_task)

        try:
            # Use appropriate timeout for complex requests
            response = await asyncio.wait_for(request_task, timeout=600.0)  # 5 minutes
            return response

        except asyncio.TimeoutError:
            print(f"⏰ Request timeout for session {session_id}")
            request_task.cancel()
            raise HTTPException(
                status_code=504,
                detail="Request timeout - operation took too long"
            )
        except asyncio.CancelledError:
            print(f"🚫 Request cancelled for session {session_id}")
            raise HTTPException(status_code=499, detail="Request cancelled")
        except Exception as e:
            print(f"❌ Request processing error for session {session_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            self._active_requests.discard(request_task)

    async def cleanup_session(self, session_id: str) -> bool:
        """Cleanup specific session"""
        async with self._main_lock:
            if session_id in self._agents:
                agent = self._agents.pop(session_id)
                self._agent_locks.pop(session_id, None)
                try:
                    await asyncio.wait_for(agent.cleanup(), timeout=15.0)
                    print(f"✅ Cleaned up session {session_id}")
                    return True
                except Exception as e:
                    print(f"⚠️ Error cleaning up session {session_id}: {e}")
                    return False
        return False

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self._agents.keys())

    async def shutdown(self):
        """Proper shutdown sequence"""
        print("🔥 Starting shutdown sequence...")

        # Cancel all active requests
        if self._active_requests:
            print(f"🚫 Cancelling {len(self._active_requests)} active requests...")
            for task in self._active_requests.copy():
                if not task.done():
                    task.cancel()

            # Wait for requests to complete/cancel
            if self._active_requests:
                try:
                    await asyncio.wait(self._active_requests, timeout=10.0)
                except asyncio.TimeoutError:
                    print("⚠️ Some requests didn't complete in time")

        # Cleanup all agents
        cleanup_tasks = []
        async with self._main_lock:
            for session_id in list(self._agents.keys()):
                agent = self._agents.pop(session_id)
                task = asyncio.create_task(agent.cleanup())
                cleanup_tasks.append(task)
            self._agent_locks.clear()

        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=20.0
                )
                print("✅ All sessions cleaned up")
            except asyncio.TimeoutError:
                print("⚠️ Some session cleanups timed out")
            except Exception as e:
                print(f"⚠️ Error during bulk cleanup: {e}")


# Global session manager
session_manager = SessionManager()


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    try:
        # Validate configuration
        config = AppConfig.from_env()
        validation_errors = config.validate()

        if validation_errors:
            print("❌ Configuration errors:")
            for error in validation_errors:
                print(f"  - {error}")
            raise RuntimeError("Invalid configuration")

        # Store config for lazy initialization
        session_manager.set_config(config)

        print("✅ BOM Agent API configuration validated")
        print("🚀 Server ready - agents will be created on demand")

        # Setup signal handlers
        def signal_handler(signum, frame):
            print(f"\n📢 Received signal {signum}, initiating graceful shutdown...")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(session_manager.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        yield

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

    # Shutdown sequence
    print("🔥 Application shutdown initiated...")
    try:
        await session_manager.shutdown()
        print("✅ Shutdown completed successfully")
    except Exception as e:
        print(f"⚠️ Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="BOM Agent API",
    description="Electronic Design BOM Management Agent - Returns structured data for UI",
    version="4.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint - returns structured data for UI consumption"""

    try:
        # Get session ID
        session_id = request.session_id or str(uuid.uuid4())

        print(f"🔥 Processing request for session {session_id}: {request.message[:100]}...")

        # Process request - get structured AgentResponse
        agent_response = await session_manager.process_request_with_timeout(
            session_id, request.message
        )

        print(f"✅ Request completed successfully for session {session_id}")

        # Convert AgentResponse to ChatResponse
        return ChatResponse(
            success=agent_response.success,
            action=agent_response.action,
            message=agent_response.message,
            data=agent_response.data,
            error=agent_response.error,
            timestamp=agent_response.timestamp,
            session_id=session_id
        )

    except HTTPException:
        raise
    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get conversation history"""

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
        print(f"❌ History endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history"""

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
        print(f"❌ Clear history error: {str(e)}")
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
        print(f"❌ Session info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
async def cleanup_session(session_id: str):
    """Cleanup specific session"""
    try:
        success = await session_manager.cleanup_session(session_id)
        if success:
            return {"message": f"Session {session_id} cleaned up successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found or cleanup failed")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Session cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""

    try:
        active_sessions = session_manager.get_active_sessions()
        return {
            "active_sessions": active_sessions,
            "total_count": len(active_sessions),
            "timestamp": datetime.now().isoformat(),
            "status": "operational"
        }
    except Exception as e:
        print(f"❌ List sessions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Fast health check"""
    try:
        active_sessions = len(session_manager.get_active_sessions())
        active_requests = len(session_manager._active_requests)

        status = "healthy"
        if active_requests > 10:
            status = "busy"

        return {
            "status": status,
            "active_sessions": active_sessions,
            "active_requests": active_requests,
            "version": "4.0.0",
            "timestamp": datetime.now().isoformat(),
            "message": f"BOM Agent API - {status}",
            "startup_mode": "lazy_initialization",
            "agent_type": "clean_react_agent"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "4.0.0"
        }


# Additional endpoint for getting component data in structured format
@app.get("/api/components/{session_id}")
async def get_components(session_id: str):
    """Get stored components in structured format for UI tables"""

    try:
        if session_id in session_manager._agents:
            agent = session_manager._agents[session_id]
            components = await agent.get_stored_components()

            # Return structured data for UI consumption
            return {
                "success": True,
                "session_id": session_id,
                "components": [comp.to_dict() for comp in components],
                "total_count": len(components),
                "enhanced_count": sum(1 for comp in components if comp.enhanced),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": True,
                "session_id": session_id,
                "components": [],
                "total_count": 0,
                "enhanced_count": 0,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        print(f"❌ Components endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        reload=False,
        access_log=True,
        log_level="info"
    )