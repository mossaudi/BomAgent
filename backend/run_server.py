# ================================================================================================
# run_server.py - Server Startup Script
# ================================================================================================

# !/usr/bin/env python3
"""
Simple server startup script with proper error handling
"""

import asyncio
import sys
import uvicorn
from pathlib import Path

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
            print("\n📝 Please check your .env file and ensure all required variables are set.")
            sys.exit(1)

        print("✅ Configuration validated successfully")
        print(f"🔧 Debug mode: {'ON' if config.debug else 'OFF'}")
        print("🚀 Starting BOM Agent server...")

        # Run server
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            reload=config.debug,
            log_level="info" if not config.debug else "debug",
            access_log=True
        )

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()