# src/services/bom_service.py
"""Modern BOM management service."""

from typing import List, Dict, Any, Optional
from datetime import datetime

from src.clients.silicon_expert_client import SiliconExpertClient
from src.services.memory_service import MemoryService
from src.core.models import BOMData
from src.core.models import ComponentData


class BOMService:
    """Service for BOM operations."""

    def __init__(self):
        self.client = SiliconExpertClient
        self.memory = MemoryService

    async def create_bom(self, name: str, description: str = "", project: str = "") -> Dict[str, Any]:
        """Create a new BOM."""
        try:
            # Create BOM via Silicon Expert API
            bom_data = {
                "name": name,
                "description": description,
                "project": project,
                "columns": ["mpn", "manufacturer", "description", "quantity", "designator"]
            }

            result = await self.client.create_bom(bom_data)

            # Store BOM info in memory
            bom = BOMData(
                name=name,
                project=project if project else None,
                metadata={
                    "description": description,
                    "api_result": result
                }
            )

            await self.memory.store_analysis_result({
                "type": "bom_created",
                "bom": bom.to_dict(),
                "timestamp": datetime.now().isoformat()
            })

            return {
                "success": True,
                "id": bom.id,
                "name": name,
                "api_response": result
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "name": name
            }

    async def add_parts(self, bom_name: str, project: str, parts: List[ComponentData]) -> Dict[str, Any]:
        """Add parts to existing BOM."""
        try:
            # Convert parts to Silicon Expert format
            se_parts = []
            for part in parts:
                se_part = {
                    "mpn": part.part_number or "Unknown",
                    "manufacturer": part.manufacturer or "Unknown",
                    "description": part.description or "",
                    "quantity": 1,
                    "designator": part.designator or ""
                }
                se_parts.append(se_part)

            # Add via Silicon Expert API
            result = await self.client.add_parts_to_bom(bom_name, project, se_parts)

            return {
                "success": True,
                "parts_count": len(se_parts),
                "bom_name": bom_name,
                "api_response": result
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "parts_count": 0
            }

    async def get_boms(self, project_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get existing BOMs."""
        try:
            result = await self.client.get_boms(project_name=project_filter or "")

            return {
                "success": True,
                "boms": result.get("boms", []),
                "projects": result.get("projects", []),
                "total_count": result.get("total_boms", 0)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "boms": []
            }

