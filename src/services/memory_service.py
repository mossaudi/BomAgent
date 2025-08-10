# src/services/memory_service.py
"""Modern memory service with clean interface."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

from src.models.state import ComponentData


class MemoryService:
    """Memory service for session-based component storage."""

    def __init__(self, session_id: Optional[str] = None, ttl_hours: int = 2):
        self.session_id = session_id or self._generate_session_id()
        self.ttl = timedelta(hours=ttl_hours)
        self._storage: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        import uuid
        return str(uuid.uuid4())

    async def store_components(self, components: List[ComponentData]) -> None:
        """Store components in session memory."""
        key = f"{self.session_id}:components"
        self._storage[key] = [asdict(comp) for comp in components]
        self._metadata[key] = {
            "timestamp": datetime.now(),
            "type": "components",
            "count": len(components)
        }

    async def get_stored_components(self) -> List[ComponentData]:
        """Retrieve stored components."""
        key = f"{self.session_id}:components"
        if key not in self._storage:
            return []

        # Check expiration
        metadata = self._metadata.get(key, {})
        if self._is_expired(metadata.get("timestamp")):
            del self._storage[key]
            del self._metadata[key]
            return []

        components_data = self._storage[key]
        return [ComponentData(**comp_data) for comp_data in components_data]

    async def store_analysis_result(self, result: Dict[str, Any]) -> None:
        """Store analysis result."""
        key = f"{self.session_id}:analysis"
        self._storage[key] = result
        self._metadata[key] = {
            "timestamp": datetime.now(),
            "type": "analysis"
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get memory status."""
        session_keys = [k for k in self._storage.keys() if k.startswith(self.session_id)]

        components_key = f"{self.session_id}:components"
        component_count = 0
        if components_key in self._storage:
            component_count = len(self._storage[components_key])

        return {
            "session_id": self.session_id,
            "keys": session_keys,
            "component_count": component_count,
            "last_activity": max(
                (meta.get("timestamp", datetime.min) for meta in self._metadata.values()),
                default=datetime.now()
            ).isoformat(),
            "usage": {
                "total_keys": len(session_keys),
                "storage_size_kb": len(str(self._storage)) // 1024
            }
        }

    def _is_expired(self, timestamp: Optional[datetime]) -> bool:
        """Check if timestamp is expired."""
        if not timestamp:
            return True
        return datetime.now() - timestamp > self.ttl

    async def cleanup(self) -> None:
        """Cleanup expired entries."""
        expired_keys = []
        for key, metadata in self._metadata.items():
            if self._is_expired(metadata.get("timestamp")):
                expired_keys.append(key)

        for key in expired_keys:
            self._storage.pop(key, None)
            self._metadata.pop(key, None)