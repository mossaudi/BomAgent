# src/clients/silicon_expert_client.py
"""Simplified Silicon Expert client with no timeouts."""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import aiohttp

from config import SiliconExpertConfig


@dataclass
class SearchResult:
    """Search result with error handling."""
    success: bool
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    lifecycle: Optional[str] = None
    confidence: float = 0.0
    raw_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class SiliconExpertClient:
    """Simplified Silicon Expert client with no timeouts."""

    def __init__(self, config: SiliconExpertConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.authenticated = False
        self.auth_expires: Optional[datetime] = None

    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if not self.session or self.session.closed:
            # NO TIMEOUT - let requests take as long as needed
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                headers={'User-Agent': 'Simple-BOM-Agent/3.0.0'}
                # NO timeout parameter - unlimited time
            )

    async def authenticate(self) -> bool:
        """Authenticate with Silicon Expert"""
        # Check if we still have valid auth
        if self.authenticated and self.auth_expires and datetime.now() < self.auth_expires:
            return True

        await self._ensure_session()

        try:
            print("🔐 Authenticating with Silicon Expert...")

            async with self.session.post(
                    f"{self.config.base_url}/search/authenticateUser",
                    headers={'content-type': 'application/x-www-form-urlencoded'},
                    data={
                        'login': self.config.username,
                        'apiKey': self.config.api_key
                    }
                    # NO timeout - let it take as long as needed
            ) as response:

                if response.status == 200:
                    self.authenticated = True
                    # Set expiration to 2 hours from now
                    self.auth_expires = datetime.now() + timedelta(hours=2)
                    print("✅ Silicon Expert authentication successful")
                    return True
                elif response.status == 401:
                    print("❌ Silicon Expert authentication failed - Invalid credentials")
                    raise Exception("Invalid credentials")
                else:
                    error_text = await response.text()
                    print(f"❌ Silicon Expert authentication failed - HTTP {response.status}")
                    raise Exception(f"Authentication failed: HTTP {response.status} - {error_text}")

        except Exception as e:
            self.authenticated = False
            self.auth_expires = None
            print(f"❌ Authentication error: {e}")
            raise

    async def search_component(self, component_data: Dict[str, Any], max_retries: int = 3) -> SearchResult:
        """Search component with unlimited time"""
        for attempt in range(max_retries):
            try:
                if not self.authenticated:
                    await self.authenticate()

                # Build search query
                search_parts = []
                for field in ['part_number', 'manufacturer', 'description', 'name']:
                    value = component_data.get(field)
                    if value and isinstance(value, str) and len(value.strip()) > 0:
                        search_parts.append(value.strip())

                if not search_parts:
                    return SearchResult(
                        success=False,
                        confidence=0.0,
                        error_message="No searchable data provided"
                    )

                search_query = ' '.join(search_parts)
                print(f"🔍 Searching Silicon Expert for: {search_query}")

                params = {
                    'fmt': 'json',
                    'pageNumber': '1',
                    'pageSize': '3',  # Reduced for faster response
                    'description': search_query
                }

                async with self.session.get(
                        f"{self.config.base_url}/search/partsearch",
                        params=params
                        # NO timeout - unlimited time
                ) as response:

                    if response.status == 401:
                        # Auth expired, retry
                        print("🔄 Authentication expired, retrying...")
                        self.authenticated = False
                        self.auth_expires = None
                        if attempt < max_retries - 1:
                            continue

                    if response.status == 429:
                        print("⏳ Rate limited, waiting...")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(10)  # Wait 10 seconds
                            continue
                        return SearchResult(
                            success=False,
                            confidence=0.0,
                            error_message="Rate limited"
                        )

                    if response.status != 200:
                        error_text = await response.text()
                        print(f"❌ Search failed - HTTP {response.status}")
                        return SearchResult(
                            success=False,
                            confidence=0.0,
                            error_message=f"Search failed: HTTP {response.status} - {error_text}"
                        )

                    data = await response.json()

                    # Parse results
                    if (data.get('Status', {}).get('Success') == 'true' and
                            data.get('Result') and len(data['Result']) > 0):

                        first_result = data['Result'][0]

                        print(f"✅ Found component: {first_result.get('PartNumber', 'Unknown')}")

                        return SearchResult(
                            success=True,
                            part_number=first_result.get('PartNumber'),
                            manufacturer=first_result.get('Manufacturer'),
                            description=first_result.get('Description'),
                            lifecycle=first_result.get('Lifecycle'),
                            raw_data=first_result
                        )
                    else:
                        print("⚠️ No matching components found")
                        return SearchResult(
                            success=True,  # Request succeeded but no results
                            confidence=0.0,
                            error_message="No matching components found"
                        )

            except Exception as e:
                print(f"⚠️ Search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Brief pause before retry
                    continue
                return SearchResult(
                    success=False,
                    confidence=0.0,
                    error_message=f"Search error: {str(e)}"
                )

        return SearchResult(
            success=False,
            confidence=0.0,
            error_message="Max retries exceeded"
        )

    async def create_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create BOM with unlimited time"""
        try:
            if not self.authenticated:
                await self.authenticate()

            print(f"📋 Creating BOM: {bom_data.get('name', 'Unknown')}")

            async with self.session.post(
                    f"{self.config.base_url}/bom/add-empty-bom",
                    headers={'Content-Type': 'application/json'},
                    json=bom_data
                    # NO timeout
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    print("✅ BOM created successfully")
                    return result
                elif response.status == 401:
                    self.authenticated = False
                    raise Exception("Authentication expired")
                elif response.status == 409:
                    raise Exception("BOM already exists")
                else:
                    error_text = await response.text()
                    raise Exception(f"BOM creation failed: HTTP {response.status} - {error_text}")

        except Exception as e:
            print(f"❌ BOM creation error: {e}")
            raise Exception(f"BOM creation error: {str(e)}")

    async def add_parts_to_bom(self, bom_name: str, project: str, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add parts to BOM with unlimited time"""
        if not parts:
            raise ValueError("No parts provided")

        try:
            if not self.authenticated:
                await self.authenticate()

            print(f"📦 Adding {len(parts)} parts to BOM: {bom_name}")

            payload = {
                "name": bom_name,
                "parentPath": project,
                "parts": parts
            }

            async with self.session.post(
                    f"{self.config.base_url}/bom/add-parts-to-bom",
                    headers={'Content-Type': 'application/json'},
                    json=payload
                    # NO timeout
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    print("✅ Parts added to BOM successfully")
                    return result
                elif response.status == 401:
                    self.authenticated = False
                    raise Exception("Authentication expired")
                elif response.status == 404:
                    raise Exception("BOM not found")
                else:
                    error_text = await response.text()
                    raise Exception(f"Add parts failed: HTTP {response.status} - {error_text}")

        except Exception as e:
            print(f"❌ Add parts error: {e}")
            raise Exception(f"Add parts error: {str(e)}")

    async def get_boms(self, project_name: str = "") -> Dict[str, Any]:
        """Get BOMs with unlimited time"""
        try:
            if not self.authenticated:
                await self.authenticate()

            print("📋 Retrieving BOMs...")

            params = {"fmt": "json"}
            if project_name:
                params["projectName"] = project_name

            async with self.session.post(
                    f"{self.config.base_url}/search/GetBOMs",
                    params=params
                    # NO timeout
            ) as response:

                if response.status == 200:
                    result = await response.json()
                    bom_count = len(result.get('boms', []))
                    print(f"✅ Retrieved {bom_count} BOMs")
                    return result
                elif response.status == 401:
                    self.authenticated = False
                    raise Exception("Authentication expired")
                else:
                    error_text = await response.text()
                    raise Exception(f"Get BOMs failed: HTTP {response.status} - {error_text}")

        except Exception as e:
            print(f"❌ Get BOMs error: {e}")
            raise Exception(f"Get BOMs error: {str(e)}")

    async def close(self):
        """Clean shutdown"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            print("✅ Silicon Expert client closed")

        self.authenticated = False
        self.auth_expires = None