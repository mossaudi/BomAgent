# src/clients/silicon_expert_client.py
"""Modern Silicon Expert client with async support."""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import aiohttp

from src.core.config import SiliconExpertConfig


@dataclass
class SearchResult:
    """Enhanced search result with error details."""
    success: bool
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    lifecycle: Optional[str] = None
    confidence: float = 0.0
    raw_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_after: Optional[int] = None


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire rate limit permit."""
        async with self._lock:
            now = datetime.now()
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls
                          if now - call_time < timedelta(minutes=1)]

            if len(self.calls) >= self.calls_per_minute:
                # Calculate wait time
                oldest_call = min(self.calls)
                wait_time = 60 - (now - oldest_call).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self.calls.append(now)


class SiliconExpertClient:
    """Enhanced Silicon Expert client with proper error handling and rate limiting."""

    def __init__(self, config: SiliconExpertConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.authenticated = False
        self.auth_expires: Optional[datetime] = None
        self.rate_limiter = RateLimiter(calls_per_minute=50)  # Conservative limit
        self._connection_timeout = aiohttp.ClientTimeout(total=30, connect=10)

    async def _ensure_session(self):
        """Ensure HTTP session exists with proper configuration."""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self._connection_timeout,
                headers={'User-Agent': 'BOM-Agent/2.1.0'}
            )

    async def authenticate(self) -> bool:
        """Enhanced authentication with expiration tracking."""
        # Check if we still have valid auth
        if self.authenticated and self.auth_expires and datetime.now() < self.auth_expires:
            return True

        await self._ensure_session()
        await self.rate_limiter.acquire()

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
                    # Set expiration to 1 hour from now (conservative)
                    self.auth_expires = datetime.now() + timedelta(hours=1)
                    return True
                elif response.status == 401:
                    raise Exception("Invalid credentials")
                elif response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', '60'))
                    raise Exception(f"Rate limited. Retry after {retry_after} seconds")
                else:
                    error_text = await response.text()
                    raise Exception(f"Authentication failed: HTTP {response.status} - {error_text}")

        except asyncio.TimeoutError:
            raise Exception("Authentication timeout")
        except aiohttp.ClientError as e:
            raise Exception(f"Authentication network error: {str(e)}")
        except Exception as e:
            self.authenticated = False
            self.auth_expires = None
            raise

    async def search_component(self, component_data: Dict[str, Any], max_retries: int = 3) -> SearchResult:
        """Enhanced component search with retry logic."""
        for attempt in range(max_retries):
            try:
                if not self.authenticated:
                    await self.authenticate()

                await self.rate_limiter.acquire()

                # Build search query with better logic
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

                    if response.status == 401:
                        # Auth expired, retry
                        self.authenticated = False
                        self.auth_expires = None
                        if attempt < max_retries - 1:
                            continue

                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', '60'))
                        if attempt < max_retries - 1:
                            await asyncio.sleep(min(retry_after, 120))  # Cap at 2 minutes
                            continue
                        return SearchResult(
                            success=False,
                            confidence=0.0,
                            error_message="Rate limited",
                            retry_after=retry_after
                        )

                    if response.status != 200:
                        error_text = await response.text()
                        return SearchResult(
                            success=False,
                            confidence=0.0,
                            error_message=f"Search failed: HTTP {response.status} - {error_text}"
                        )

                    data = await response.json()

                    # Enhanced result parsing
                    if (data.get('Status', {}).get('Success') == 'true' and
                            data.get('Result') and len(data['Result']) > 0):

                        first_result = data['Result'][0]
                        confidence = self._calculate_confidence(first_result, component_data)

                        return SearchResult(
                            success=True,
                            part_number=first_result.get('PartNumber'),
                            manufacturer=first_result.get('Manufacturer'),
                            description=first_result.get('Description'),
                            lifecycle=first_result.get('Lifecycle'),
                            confidence=confidence,
                            raw_data=first_result
                        )
                    else:
                        return SearchResult(
                            success=True,  # Request succeeded but no results
                            confidence=0.0,
                            error_message="No matching components found"
                        )

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return SearchResult(
                    success=False,
                    confidence=0.0,
                    error_message="Request timeout"
                )

            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return SearchResult(
                    success=False,
                    confidence=0.0,
                    error_message=f"Network error: {str(e)}"
                )

            except Exception as e:
                return SearchResult(
                    success=False,
                    confidence=0.0,
                    error_message=f"Unexpected error: {str(e)}"
                )

        return SearchResult(
            success=False,
            confidence=0.0,
            error_message="Max retries exceeded"
        )

    def _calculate_confidence(self, result: Dict[str, Any], original_data: Dict[str, Any]) -> float:
        """Enhanced confidence calculation."""
        base_confidence = {
            'Exact': 1.0,
            'High': 0.9,
            'Medium': 0.7,
            'Low': 0.5
        }.get(result.get('MatchRating', ''), 0.3)

        # Boost confidence if key fields match
        confidence_boost = 0.0

        # Check part number match
        original_pn = original_data.get('part_number', '').lower()
        result_pn = result.get('PartNumber', '').lower()
        if original_pn and result_pn and original_pn in result_pn:
            confidence_boost += 0.1

        # Check manufacturer match
        original_mfr = original_data.get('manufacturer', '').lower()
        result_mfr = result.get('Manufacturer', '').lower()
        if original_mfr and result_mfr and original_mfr in result_mfr:
            confidence_boost += 0.1

        return min(1.0, base_confidence + confidence_boost)

    async def create_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced BOM creation with proper error handling."""
        try:
            if not self.authenticated:
                await self.authenticate()

            await self.rate_limiter.acquire()

            async with self.session.post(
                    f"{self.config.base_url}/bom/add-empty-bom",
                    headers={'Content-Type': 'application/json'},
                    json=bom_data
            ) as response:

                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    self.authenticated = False
                    raise Exception("Authentication expired")
                elif response.status == 409:
                    raise Exception("BOM already exists")
                else:
                    error_text = await response.text()
                    raise Exception(f"BOM creation failed: HTTP {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"BOM creation error: {str(e)}")

    async def add_parts_to_bom(self, bom_name: str, project: str, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced parts addition with validation."""
        if not parts:
            raise ValueError("No parts provided")

        try:
            if not self.authenticated:
                await self.authenticate()

            await self.rate_limiter.acquire()

            payload = {
                "name": bom_name,
                "parentPath": project,
                "parts": parts
            }

            async with self.session.post(
                    f"{self.config.base_url}/bom/add-parts-to-bom",
                    headers={'Content-Type': 'application/json'},
                    json=payload
            ) as response:

                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    self.authenticated = False
                    raise Exception("Authentication expired")
                elif response.status == 404:
                    raise Exception("BOM not found")
                else:
                    error_text = await response.text()
                    raise Exception(f"Add parts failed: HTTP {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"Add parts error: {str(e)}")

    async def get_boms(self, project_name: str = "") -> Dict[str, Any]:
        """Enhanced BOM retrieval."""
        try:
            if not self.authenticated:
                await self.authenticate()

            await self.rate_limiter.acquire()

            params = {"fmt": "json"}
            if project_name:
                params["projectName"] = project_name

            async with self.session.post(
                    f"{self.config.base_url}/search/GetBOMs",
                    params=params
            ) as response:

                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    self.authenticated = False
                    raise Exception("Authentication expired")
                else:
                    error_text = await response.text()
                    raise Exception(f"Get BOMs failed: HTTP {response.status} - {error_text}")

        except Exception as e:
            raise Exception(f"Get BOMs error: {str(e)}")

    async def close(self):
        """Enhanced cleanup."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

        self.authenticated = False
        self.auth_expires = None