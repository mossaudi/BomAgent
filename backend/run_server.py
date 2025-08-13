# run_fixed_server.py - Startup Script for Fixed Server
"""
Startup script for the fixed server with lazy initialization.
"""

import sys
from pathlib import Path

import uvicorn

from config import AppConfig

# Add project root to path
project_root = Path(__file__).parent
src_dir = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def main():
    """Main entry point"""
    try:
        config = AppConfig.from_env()
        validation_errors = config.validate()

        if validation_errors:
            print("❌ Configuration errors:")
            for error in validation_errors:
                print(f"  - {error}")
            print("\n📋 Please check your .env file and ensure all required variables are set.")
            sys.exit(1)

        print("✅ Configuration validated successfully")
        print(f"🔧 Debug mode: {'ON' if config.debug else 'OFF'}")
        print(f"🤖 LLM Provider: {config.llm_provider}")
        print("🚀 Starting BOM Agent server with lazy initialization...")
        print("")
        print("🔥 Key improvements:")
        print("  - Agents created on-demand (no startup blocking)")
        print("  - Fast health checks (< 2 seconds)")
        print("  - Better timeout handling")
        print("  - Improved error messages")
        print("")

        # Use the fixed server module
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=9000,
            reload=config.debug,
            log_level="info" if not config.debug else "debug",
            access_log=True,
            timeout_keep_alive=30,
            timeout_graceful_shutdown=15
        )

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()