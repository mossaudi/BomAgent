# services/memory.py
"""Session-based memory management system for multi-user support."""

import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List


@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    key: str
    value: Any
    timestamp: float
    expiry: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the memory entry has expired."""
        if self.expiry is None:
            return False
        return time.time() > self.expiry

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'timestamp': self.timestamp,
            'expiry': self.expiry,
            'metadata': self.metadata
        }


class MemoryBackend(ABC):
    """Abstract base class for memory backends."""

    @abstractmethod
    def get(self, session_id: str, key: str) -> Optional[MemoryEntry]:
        """Get a memory entry."""
        pass

    @abstractmethod
    def set(self, session_id: str, entry: MemoryEntry) -> None:
        """Set a memory entry."""
        pass

    @abstractmethod
    def delete(self, session_id: str, key: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    def list_keys(self, session_id: str) -> List[str]:
        """List all keys for a session."""
        pass

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Clear all entries for a session."""
        pass


class InMemoryBackend(MemoryBackend):
    """In-memory backend with thread safety."""

    def __init__(self):
        self._storage: Dict[str, Dict[str, MemoryEntry]] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str, key: str) -> Optional[MemoryEntry]:
        with self._lock:
            session_data = self._storage.get(session_id, {})
            entry = session_data.get(key)

            if entry and entry.is_expired:
                del session_data[key]
                return None

            return entry

    def set(self, session_id: str, entry: MemoryEntry) -> None:
        with self._lock:
            if session_id not in self._storage:
                self._storage[session_id] = {}
            self._storage[session_id][entry.key] = entry

    def delete(self, session_id: str, key: str) -> bool:
        with self._lock:
            session_data = self._storage.get(session_id, {})
            if key in session_data:
                del session_data[key]
                return True
            return False

    def list_keys(self, session_id: str) -> List[str]:
        with self._lock:
            session_data = self._storage.get(session_id, {})
            # Clean up expired entries and return valid keys
            valid_keys = []
            expired_keys = []

            for key, entry in session_data.items():
                if entry.is_expired:
                    expired_keys.append(key)
                else:
                    valid_keys.append(key)

            # Clean up expired entries
            for key in expired_keys:
                del session_data[key]

            return valid_keys

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._storage:
                del self._storage[session_id]


class SessionMemoryManager:
    """Session-based memory manager for multi-user support."""

    def __init__(self, backend: MemoryBackend = None, default_ttl: int = 3600):
        self.backend = backend or InMemoryBackend()
        self.default_ttl = default_ttl
        self._current_session_id: Optional[str] = None

    def create_session(self, session_id: str = None) -> str:
        """Create or set a session ID."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        self._current_session_id = session_id
        return session_id

    def set_session(self, session_id: str) -> None:
        """Set the current session ID."""
        self._current_session_id = session_id

    def get_session(self) -> Optional[str]:
        """Get the current session ID."""
        return self._current_session_id

    def store(self, key: str, value: Any, ttl: Optional[int] = None,
              metadata: Dict[str, Any] = None, session_id: str = None) -> None:
        """Store a value in memory."""
        if session_id is None:
            session_id = self._current_session_id

        if session_id is None:
            raise ValueError("No session ID provided and no current session set")

        expiry = None
        if ttl is not None:
            expiry = time.time() + ttl
        elif self.default_ttl > 0:
            expiry = time.time() + self.default_ttl

        entry = MemoryEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            expiry=expiry,
            metadata=metadata or {}
        )

        self.backend.set(session_id, entry)

    def retrieve(self, key: str, session_id: str = None) -> Optional[Any]:
        """Retrieve a value from memory."""
        if session_id is None:
            session_id = self._current_session_id

        if session_id is None:
            return None

        entry = self.backend.get(session_id, key)
        return entry.value if entry else None

    def retrieve_with_metadata(self, key: str, session_id: str = None) -> Optional[MemoryEntry]:
        """Retrieve a memory entry with full metadata."""
        if session_id is None:
            session_id = self._current_session_id

        if session_id is None:
            return None

        return self.backend.get(session_id, key)

    def delete(self, key: str, session_id: str = None) -> bool:
        """Delete a memory entry."""
        if session_id is None:
            session_id = self._current_session_id

        if session_id is None:
            return False

        return self.backend.delete(session_id, key)

    def list_keys(self, session_id: str = None) -> List[str]:
        """List all keys in current session."""
        if session_id is None:
            session_id = self._current_session_id

        if session_id is None:
            return []

        return self.backend.list_keys(session_id)

    def clear_session(self, session_id: str = None) -> None:
        """Clear all entries for a session."""
        if session_id is None:
            session_id = self._current_session_id

        if session_id is None:
            return

        self.backend.clear_session(session_id)

    def get_session_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get a summary of the current session's memory."""
        keys = self.list_keys(session_id)
        summary = {
            'session_id': session_id or self._current_session_id,
            'total_keys': len(keys),
            'keys': keys,
            'created_at': time.time()
        }

        # Add key details
        key_details = {}
        for key in keys:
            entry = self.retrieve_with_metadata(key, session_id)
            if entry:
                key_details[key] = {
                    'timestamp': entry.timestamp,
                    'has_expiry': entry.expiry is not None,
                    'metadata': entry.metadata
                }

        summary['key_details'] = key_details
        return summary


# Specialized memory helpers for common operations
class ComponentMemoryHelper:
    """Helper class for component-specific memory operations."""

    def __init__(self, memory_manager: SessionMemoryManager):
        self.memory = memory_manager

    def store_analysis_result(self, search_result, image_url: str = None) -> None:
        """Store a complete analysis result."""
        self.memory.store('last_search_result', search_result, metadata={
            'type': 'search_result',
            'image_url': image_url,
            'component_count': len(search_result.components) if hasattr(search_result, 'components') else 0
        })

        if hasattr(search_result, 'components'):
            self.memory.store('last_components', search_result.components, metadata={
                'type': 'component_list',
                'source': 'analysis'
            })

    def get_last_components(self):
        """Get the last analyzed components."""
        return self.memory.retrieve('last_components')

    def get_last_search_result(self):
        """Get the last search result."""
        return self.memory.retrieve('last_search_result')

    def has_components(self) -> bool:
        """Check if components are available in memory."""
        return self.memory.retrieve('last_components') is not None

    def get_component_summary(self) -> Dict[str, Any]:
        """Get a summary of stored components."""
        components = self.get_last_components()
        search_result = self.get_last_search_result()

        if not components:
            return {'has_components': False}

        summary = {
            'has_components': True,
            'component_count': len(components),
            'success_rate': getattr(search_result, 'success_rate', 0) if search_result else 0
        }

        # Add component names
        if hasattr(components, '__iter__'):
            try:
                summary['component_names'] = [
                    getattr(comp, 'name', str(comp)) for comp in components[:10]  # First 10
                ]
            except:
                summary['component_names'] = ['(unable to extract names)']

        return summary


# Global memory manager instance
_memory_manager = SessionMemoryManager()
_component_helper = ComponentMemoryHelper(_memory_manager)


def get_memory_manager() -> SessionMemoryManager:
    """Get the global memory manager instance."""
    return _memory_manager


def get_component_memory() -> ComponentMemoryHelper:
    """Get the component memory helper."""
    return _component_helper


def configure_memory_manager(backend: MemoryBackend = None, default_ttl: int = 3600):
    """Configure the global memory manager."""
    global _memory_manager, _component_helper
    _memory_manager = SessionMemoryManager(backend, default_ttl)
    _component_helper = ComponentMemoryHelper(_memory_manager)