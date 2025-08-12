#!/usr/bin/env python3
"""
Run script for the BOM Agent server.
Handles proper path setup and server initialization.
"""

import os
import sys
from pathlib import Path

from src.core.config import AppConfig

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.absolute()
src_dir = backend_dir / "src"

# Add both directories to path
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


# Now we can import and run the server
if __name__ == "__main__":
    import uvicorn
    from server import app

    config = AppConfig.from_env()
    validation_errors = config.validate()
    if validation_errors:
        print("❌ Configuration errors:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)

    print(f"Starting BOM Agent server from: {backend_dir}")
    print(f"Python path includes: {[p for p in sys.path if 'backend' in p or 'BomAgent' in p]}")

    # Run the server
    uvicorn.run(
        'server:app',
        host="0.0.0.0",
        port=8000,
        reload=False,
        reload_dirs=[str(backend_dir)]
    )