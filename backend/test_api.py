# test_api_improved.py - Improved API Tests for Lazy Initialization
"""
Improved test suite that accounts for:
1. Lazy agent initialization
2. Longer timeouts for first-time agent creation
3. Better error handling
4. Progressive test structure
"""

import pytest
import aiohttp
import asyncio
import json
import time
from typing import Optional

# Test configuration
BASE_URL = "http://localhost:9000"
SHORT_TIMEOUT = 15  # For fast operations like health checks
MEDIUM_TIMEOUT = 45  # For agent initialization
LONG_TIMEOUT = 120  # For complex operations like analysis


class TestBOMAgentAPIImproved:
    """Improved BOM Agent API test suite accounting for lazy initialization"""

    def __init__(self):
        self.session_id: Optional[str] = None

    @pytest.mark.asyncio
    async def test_01_health_endpoint(self):
        """Test health endpoint - should be fast with lazy initialization"""
        async with aiohttp.ClientSession() as session:
            start_time = time.time()

            async with session.get(
                    f"{BASE_URL}/api/health",
                    timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
            ) as response:
                duration = time.time() - start_time

                assert response.status == 200, f"Health check failed with status {response.status}"

                data = await response.json()
                assert "status" in data, "Health response missing status field"
                assert "startup_mode" in data, "Health response missing startup_mode field"

                print(f"✅ Health check passed in {duration:.2f}s: {data['status']}")
                print(f"🔧 Startup mode: {data.get('startup_mode', 'unknown')}")

    @pytest.mark.asyncio
    async def test_02_first_chat_request(self):
        """Test first chat request - will trigger agent initialization"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "message": "help",
                "session_id": None
            }

            print("📤 Sending first help request (will initialize agent)...")
            start_time = time.time()

            # Use longer timeout for first request due to agent initialization
            async with session.post(
                    f"{BASE_URL}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=MEDIUM_TIMEOUT)
            ) as response:
                duration = time.time() - start_time
                print(f"⏱️ First request time: {duration:.2f}s")

                assert response.status == 200, f"Chat help failed with status {response.status}"

                data = await response.json()
                assert "success" in data, "Response missing success field"
                assert data["success"] is True, "Help request was not successful"

                # Store session ID for subsequent tests
                if "data" in data and "session_id" in data["data"]:
                    self.session_id = data["data"]["session_id"]

                assert "message" in data, "Response missing message field"
                assert len(data["message"]) > 0, "Help message is empty"

                print(f"✅ First chat request successful, session: {self.session_id}")
                print(f"📄 Message length: {len(data['message'])}")

    @pytest.mark.asyncio
    async def test_03_subsequent_chat_request(self):
        """Test subsequent chat request - should be faster with existing agent"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "message": "help",
                "session_id": self.session_id  # Reuse existing session
            }

            print("📤 Sending subsequent help request (reusing agent)...")
            start_time = time.time()

            async with session.post(
                    f"{BASE_URL}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
            ) as response:
                duration = time.time() - start_time
                print(f"⏱️ Subsequent request time: {duration:.2f}s")

                assert response.status == 200, f"Subsequent chat failed with status {response.status}"

                data = await response.json()
                assert data["success"] is True, "Subsequent help request was not successful"

                print(f"✅ Subsequent chat request successful - much faster!")

    @pytest.mark.asyncio
    async def test_04_invalid_url_handling(self):
        """Test schematic analysis with invalid URL"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "message": "analyze_schematic(invalid-url)",
                "session_id": self.session_id
            }

            print("📤 Sending invalid URL request...")
            start_time = time.time()

            async with session.post(
                    f"{BASE_URL}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
            ) as response:
                duration = time.time() - start_time
                print(f"⏱️ Response time: {duration:.2f}s")

                assert response.status == 200, f"Invalid URL test failed with status {response.status}"

                data = await response.json()
                message = data.get("message", "")

                # Check that invalid URL is properly handled
                invalid_url_indicators = [
                    "Invalid URL",
                    "invalid",
                    "valid HTTP/HTTPS URL",
                    "URL Required"
                ]

                assert any(indicator in message for indicator in invalid_url_indicators), \
                    f"Invalid URL not properly handled. Response: {message[:200]}..."

                print("✅ Invalid URL properly handled")

    @pytest.mark.asyncio
    async def test_05_show_empty_components(self):
        """Test showing components when none exist"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "message": "show_components_table()",
                "session_id": self.session_id
            }

            print("📤 Sending show components request...")
            start_time = time.time()

            async with session.post(
                    f"{BASE_URL}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
            ) as response:
                duration = time.time() - start_time
                print(f"⏱️ Response time: {duration:.2f}s")

                assert response.status == 200, f"Show components test failed with status {response.status}"

                data = await response.json()
                message = data.get("message", "")

                # Check that empty components is properly handled
                empty_indicators = [
                    "No components available",
                    "Analyze a schematic first",
                    "analyze_schematic(URL)"
                ]

                assert any(indicator in message for indicator in empty_indicators), \
                    f"Empty components not properly handled. Response: {message[:200]}..."

                print("✅ Empty components properly handled")

    @pytest.mark.asyncio
    async def test_06_sessions_endpoint(self):
        """Test sessions list endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{BASE_URL}/api/sessions",
                    timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
            ) as response:
                assert response.status == 200, f"Sessions endpoint failed with status {response.status}"

                data = await response.json()
                assert "active_sessions" in data, "Sessions response missing active_sessions"
                assert "total_count" in data, "Sessions response missing total_count"

                session_count = data.get("total_count", 0)
                print(f"✅ Sessions endpoint works: {session_count} active sessions")

    @pytest.mark.asyncio
    async def test_07_concurrent_new_sessions(self):
        """Test multiple concurrent requests with new sessions"""
        async with aiohttp.ClientSession() as session:
            # Create multiple help requests concurrently with new sessions
            tasks = []
            for i in range(3):
                payload = {
                    "message": "help",
                    "session_id": f"test_concurrent_{i}_{int(time.time())}"
                }
                task = session.post(
                    f"{BASE_URL}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=MEDIUM_TIMEOUT)  # Longer timeout for new agents
                )
                tasks.append(task)

            print("📤 Sending 3 concurrent requests with new sessions...")
            start_time = time.time()

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            print(f"⏱️ Concurrent requests time: {duration:.2f}s")

            success_count = 0
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    print(f"❌ Request {i + 1} failed: {response}")
                    pytest.fail(f"Concurrent request {i + 1} failed: {response}")
                else:
                    assert response.status == 200, f"Request {i + 1} failed with status {response.status}"
                    success_count += 1
                    print(f"✅ Request {i + 1} succeeded")
                    response.close()

            assert success_count == 3, f"Only {success_count}/3 concurrent requests succeeded"
            print("✅ All concurrent requests succeeded")

    @pytest.mark.asyncio
    async def test_08_history_endpoint(self):
        """Test history endpoint with existing session"""
        if not self.session_id:
            pytest.skip("No session ID available for history test")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{BASE_URL}/api/history/{self.session_id}",
                    timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
            ) as response:
                assert response.status == 200, f"History endpoint failed with status {response.status}"

                data = await response.json()
                assert "session_id" in data, "History response missing session_id"
                assert "history" in data, "History response missing history"
                assert "total_count" in data, "History response missing total_count"

                print(f"✅ History endpoint works: {data['total_count']} messages")

    @pytest.mark.asyncio
    async def test_09_session_info_endpoint(self):
        """Test session info endpoint"""
        if not self.session_id:
            pytest.skip("No session ID available for session info test")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{BASE_URL}/api/sessions/{self.session_id}",
                    timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
            ) as response:
                assert response.status == 200, f"Session info failed with status {response.status}"

                data = await response.json()
                assert "session_id" in data, "Session info missing session_id"
                assert "history_count" in data, "Session info missing history_count"
                assert "initialized" in data, "Session info missing initialized"

                print(f"✅ Session info works: initialized={data['initialized']}")

    @pytest.mark.asyncio
    async def test_10_error_handling(self):
        """Test error handling with malformed request"""
        async with aiohttp.ClientSession() as session:
            # Send request with missing message
            payload = {
                "session_id": "test_error"
                # Missing required 'message' field
            }

            async with session.post(
                    f"{BASE_URL}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
            ) as response:
                # Should return 422 (validation error) or 400 (bad request)
                assert response.status in [400, 422], \
                    f"Expected validation error, got status {response.status}"

                print("✅ Error handling works correctly")

    @pytest.mark.asyncio
    async def test_11_timeout_behavior(self):
        """Test that health check is consistently fast"""
        async with aiohttp.ClientSession() as session:
            times = []
            for i in range(5):
                start_time = time.time()

                async with session.get(
                        f"{BASE_URL}/api/health",
                        timeout=aiohttp.ClientTimeout(total=5)  # Very short timeout
                ) as response:
                    duration = time.time() - start_time
                    times.append(duration)

                    assert response.status == 200, f"Health check {i + 1} failed"

                await asyncio.sleep(0.1)  # Small delay between requests

            avg_time = sum(times) / len(times)
            max_time = max(times)

            print(f"✅ Health checks: avg={avg_time:.3f}s, max={max_time:.3f}s")

            # Health checks should be consistently fast with lazy initialization
            assert max_time < 2.0, f"Health check too slow: {max_time:.3f}s"

    @pytest.mark.asyncio
    async def test_12_agent_reuse_performance(self):
        """Test that subsequent requests to same session are fast"""
        if not self.session_id:
            pytest.skip("No session ID available for performance test")

        async with aiohttp.ClientSession() as session:
            times = []
            for i in range(3):
                payload = {
                    "message": "help",
                    "session_id": self.session_id
                }

                start_time = time.time()

                async with session.post(
                        f"{BASE_URL}/api/chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=SHORT_TIMEOUT)
                ) as response:
                    duration = time.time() - start_time
                    times.append(duration)

                    assert response.status == 200, f"Reuse request {i + 1} failed"

                    data = await response.json()
                    assert data["success"] is True, f"Reuse request {i + 1} unsuccessful"

                await asyncio.sleep(0.1)  # Small delay between requests

            avg_time = sum(times) / len(times)
            max_time = max(times)

            print(f"✅ Agent reuse: avg={avg_time:.3f}s, max={max_time:.3f}s")

            # Subsequent requests should be much faster than initial agent creation
            assert max_time < 10.0, f"Agent reuse too slow: {max_time:.3f}s"


# Pytest fixtures and configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def test_server_is_running():
    """Quick synchronous test to verify server is accessible"""
    import requests
    try:
        print("🔍 Testing server accessibility...")
        response = requests.get(f"{BASE_URL}/api/health", timeout=15)
        assert response.status_code == 200, f"Server not responding. Status: {response.status_code}"

        data = response.json()
        print(f"✅ Server is running: {data.get('status', 'unknown')}")
        print(f"🔧 Startup mode: {data.get('startup_mode', 'unknown')}")

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Server is not running or not accessible: {e}")


# Test order configuration
pytest_plugins = []

if __name__ == "__main__":
    # Allow running this file directly
    print("🧪 Running BOM Agent API Tests (Improved)")
    print("=" * 50)
    print("Make sure your server is running on http://localhost:8000")
    print("Use: python -m pytest test_api_improved.py -v --tb=short")
    print("")
    print("🔧 Test Features:")
    print("- Accounts for lazy agent initialization")
    print("- Progressive timeouts (short/medium/long)")
    print("- Performance testing for agent reuse")
    print("- Better error handling and reporting")
    print("")
    print("⏱️ Expected timings:")
    print("- Health checks: < 2 seconds")
    print("- First chat request: 15-45 seconds (agent initialization)")
    print("- Subsequent requests: < 10 seconds (agent reuse)")