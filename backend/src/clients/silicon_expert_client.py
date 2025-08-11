# src/clients/silicon_expert_client.py
"""Modern Silicon Expert client with async support."""

import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.core.config import SiliconExpertConfig


@dataclass
class SearchResult:
    """Silicon Expert search result."""
    success: bool
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    lifecycle: Optional[str] = None
    confidence: float = 0.0
    raw_data: Dict[str, Any] = None


class SiliconExpertClient:
    """Modern async Silicon Expert API client."""

    def __init__(self, config: SiliconExpertConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.authenticated = False

    async def _ensure_session(self):
        """Ensure HTTP session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def authenticate(self) -> bool:
        """Authenticate with Silicon Expert API."""
        await self._ensure_session()

        try:
            async with self.session.post(
                    f"{self.config.base_url}/search/authenticateUser",
                    headers={'content-type': 'application/x-www-form-urlencoded'},
                    data={
                        'login': self.config.username,
                        'apiKey': self.config.api_key
                    }
            ) as response:

                if response.status == 200:
                    self.authenticated = True
                    return True
                else:
                    raise Exception(f"Authentication failed: HTTP {response.status}")

        except Exception as e:
            raise Exception(f"Authentication error: {str(e)}")

    async def search_component(self, component_data: Dict[str, Any]) -> SearchResult:
        """Search for a single component."""
        if not self.authenticated:
            await self.authenticate()

        # Build search query
        search_parts = []
        for field in ['part_number', 'manufacturer', 'description']:
            value = component_data.get(field)
            if value:
                search_parts.append(str(value))

        if not search_parts:
            return SearchResult(success=False, confidence=0.0)

        search_query = ' '.join(search_parts)

        try:
            params = {
                'fmt': 'json',
                'pageNumber': '1',
                'pageSize': '5',
                'description': search_query
            }

            async with self.session.get(
                    f"{self.config.base_url}/search/partsearch",
                    params=params
            ) as response:

                if response.status != 200:
                    return SearchResult(success=False, confidence=0.0)

                data = await response.json()

                # Parse results
                if (data.get('Status', {}).get('Success') == 'true' and
                        data.get('Result') and len(data['Result']) > 0):

                    first_result = data['Result'][0]

                    return SearchResult(
                        success=True,
                        part_number=first_result.get('PartNumber'),
                        manufacturer=first_result.get('Manufacturer'),
                        description=first_result.get('Description'),
                        lifecycle=first_result.get('Lifecycle'),
                        confidence=self._calculate_confidence(first_result),
                        raw_data=first_result
                    )
                else:
                    return SearchResult(success=False, confidence=0.0)

        except Exception as e:
            return SearchResult(success=False, confidence=0.0)

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score based on match rating."""
        match_rating = result.get('MatchRating', '')

        confidence_map = {
            'Exact': 1.0,
            'High': 0.9,
            'Medium': 0.7,
            'Low': 0.5
        }

        return confidence_map.get(match_rating, 0.3)

    async def create_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new BOM."""
        if not self.authenticated:
            await self.authenticate()

        try:
            async with self.session.post(
                    f"{self.config.base_url}/bom/add-empty-bom",
                    headers={'Content-Type': 'application/json'},
                    json=bom_data
            ) as response:

                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"BOM creation failed: HTTP {response.status}")

        except Exception as e:
            raise Exception(f"BOM creation error: {str(e)}")

    async def add_parts_to_bom(self, bom_name: str, project: str, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add parts to existing BOM."""
        if not self.authenticated:
            await self.authenticate()

        payload = {
            "name": bom_name,
            "parentPath": project,
            "parts": parts
        }

        try:
            async with self.session.post(
                    f"{self.config.base_url}/bom/add-parts-to-bom",
                    headers={'Content-Type': 'application/json'},
                    json=payload
            ) as response:

                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Add parts failed: HTTP {response.status}")

        except Exception as e:
            raise Exception(f"Add parts error: {str(e)}")

    async def get_boms(self, project_name: str = "") -> Dict[str, Any]:
        """Get existing BOMs."""
        if not self.authenticated:
            await self.authenticate()

        params = {"fmt": "json"}
        if project_name:
            params["projectName"] = project_name

        try:
            async with self.session.post(
                    f"{self.config.base_url}/search/GetBOMs",
                    params=params
            ) as response:

                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Get BOMs failed: HTTP {response.status}")

        except Exception as e:
            raise Exception(f"Get BOMs error: {str(e)}")

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None