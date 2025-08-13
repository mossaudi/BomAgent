# src/services/memory_service.py
"""Modern memory service with clean interface."""

import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.core.models import ComponentData


class MemoryService:
    """Enhanced memory service with persistence and cleanup."""

    def __init__(self, session_id: Optional[str] = None, ttl_hours: int = 2):
        self.session_id = session_id or self._generate_session_id()
        self.ttl = timedelta(hours=ttl_hours)
        self._storage: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup()

    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        import uuid
        return str(uuid.uuid4())

    def _start_cleanup(self):
        """Start background cleanup task."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self.cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup error: {e}")

    async def store_components(self, components: List[ComponentData]) -> None:
        """Thread-safe component storage."""
        with self._lock:
            key = f"{self.session_id}:components"
            serialized_data = []

            for comp in components:
                try:
                    serialized_data.append(asdict(comp))
                except Exception as e:
                    print(f"Serialization error for component: {e}")
                    continue

            self._storage[key] = serialized_data
            self._metadata[key] = {
                "timestamp": datetime.now(),
                "type": "components",
                "count": len(serialized_data)
            }

    async def get_stored_components(self) -> List[ComponentData]:
        """Thread-safe component retrieval."""
        async with self._lock:
            key = f"{self.session_id}:components"
            if key not in self._storage:
                return []

            metadata = self._metadata.get(key, {})
            if self._is_expired(metadata.get("timestamp")):
                del self._storage[key]
                del self._metadata[key]
                return []

            try:
                components_data = self._storage[key]
                return [ComponentData(**comp_data) for comp_data in components_data]
            except Exception as e:
                print(f"Deserialization error: {e}")
                return []

    async def store_analysis_result(self, result: Dict[str, Any]) -> None:
        """Thread-safe analysis result storage."""
        with self._lock:
            key = f"{self.session_id}:analysis"
            self._storage[key] = result
            self._metadata[key] = {
                "timestamp": datetime.now(),
                "type": "analysis"
            }

    async def get_status(self) -> Dict[str, Any]:
        """Get memory status (no sensitive data exposed)."""
        with self._lock:
            session_keys = [k for k in self._storage.keys() if k.startswith(self.session_id)]

            components_key = f"{self.session_id}:components"
            component_count = 0
            if components_key in self._storage:
                component_count = len(self._storage[components_key])

            return {
                "session_id": self.session_id,
                "component_count": component_count,
                "last_activity": max(
                    (meta.get("timestamp", datetime.min) for meta in self._metadata.values()),
                    default=datetime.now()
                ).isoformat(),
                "usage": {
                    "active_keys": len(session_keys),
                    "storage_size_kb": len(str(self._storage)) // 1024
                }
            }

    def _is_expired(self, timestamp: Optional[datetime]) -> bool:
        """Check if timestamp is expired."""
        if not timestamp:
            return True
        return datetime.now() - timestamp > self.ttl

    async def cleanup(self) -> int:
        """Thread-safe cleanup of expired entries."""
        with self._lock:
            expired_keys = []
            for key, metadata in self._metadata.items():
                if self._is_expired(metadata.get("timestamp")):
                    expired_keys.append(key)

            for key in expired_keys:
                self._storage.pop(key, None)
                self._metadata.pop(key, None)

            return len(expired_keys)

    def __del__(self):
        """Cleanup on destruction."""
        if self._cleanup_task:
            self._cleanup_task.cancel()