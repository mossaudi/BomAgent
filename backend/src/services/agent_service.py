"""Service layer for orchestrating agent operations."""

from typing import Dict, Any, List
from src.core.container import Container
from src.models.state import ComponentData, BOMData, ResponseBuilder


class AgentOrchestrator:
    """Orchestrates complex agent operations with clean separation of concerns."""

    def __init__(self, container: Container):
        self.container = container

    async def analyze_schematic(self, image_url: str) -> Dict[str, Any]:
        """Orchestrate schematic analysis workflow."""
        try:
            # Initialize container if needed
            await self.container.initialize()

            # Analyze schematic
            schematic_service = self.container.get_schematic_service()
            raw_analysis = await schematic_service.analyze(image_url)

            # Parse components
            component_service = self.container.get_component_service()
            components = await component_service.parse_components(raw_analysis)

            # Store in memory for future use
            memory_service = self.container.get_memory_service()
            await memory_service.store_components(components)

            # Convert to standardized format
            component_data = [
                ComponentData(
                    name=comp.get('name', ''),
                    part_number=comp.get('part_number'),
                    manufacturer=comp.get('manufacturer'),
                    description=comp.get('description'),
                    value=comp.get('value'),
                    designator=comp.get('designator'),
                    confidence=comp.get('confidence', 0.8)
                ) for comp in components
            ]

            return {
                "success": True,
                "components": [comp.to_dict() for comp in component_data],
                "total_components": len(component_data),
                "image_url": image_url,
                "confidence": sum(comp.confidence for comp in component_data) / len(
                    component_data) if component_data else 0
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "components": [],
                "total_components": 0
            }

    async def search_components(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Orchestrate component search workflow."""
        try:
            await self.container.initialize()

            component_service = self.container.get_component_service()
            enhanced_components = await component_service.search_and_enhance(components)

            return {
                "success": True,
                "components": [comp.to_dict() for comp in enhanced_components],
                "total_searched": len(components),
                "enhanced_count": len([c for c in enhanced_components if c.metadata.get('enhanced')])
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "components": []
            }

    async def create_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate BOM creation workflow."""
        try:
            await self.container.initialize()

            bom_service = self.container.get_bom_service()
            result = await bom_service.create_bom(
                name=bom_data.get('name', ''),
                description=bom_data.get('description', ''),
                project=bom_data.get('project', '')
            )

            return {
                "success": True,
                "bom_id": result.get('id'),
                "name": bom_data.get('name'),
                "message": f"BOM '{bom_data.get('name')}' created successfully"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "bom_id": None
            }

    async def add_parts_to_bom(self, parts_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate adding parts to BOM workflow."""
        try:
            await self.container.initialize()

            bom_service = self.container.get_bom_service()
            memory_service = self.container.get_memory_service()

            # Get parts from memory if not provided
            parts = parts_data.get('parts', [])
            if not parts:
                stored_components = await memory_service.get_stored_components()
                parts = [comp.to_dict() for comp in stored_components]

            result = await bom_service.add_parts(
                bom_name=parts_data.get('bom_name', ''),
                project=parts_data.get('project', ''),
                parts=parts
            )

            return {
                "success": True,
                "parts_added": len(parts),
                "bom_name": parts_data.get('bom_name'),
                "message": f"Added {len(parts)} parts to BOM"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "parts_added": 0
            }

    async def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        try:
            await self.container.initialize()

            memory_service = self.container.get_memory_service()
            status = await memory_service.get_status()

            return {
                "success": True,
                "session_id": self.container.session_id,
                "stored_components": status.get('component_count', 0),
                "memory_keys": status.get('keys', []),
                "last_activity": status.get('last_activity'),
                "storage_usage": status.get('usage', {})
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "session_id": self.container.session_id
            }

