# src/services/component_service.py
"""Modern component search and enhancement service."""

from typing import List, Dict, Any

from backend.src.clients.silicon_expert_client import SiliconExpertClient
from backend.src.models.state import ComponentData
from backend.src.services.memory_service import MemoryService


class ComponentService:
    """Service for component operations."""

    def __init__(self, silicon_expert_client: SiliconExpertClient, memory_service: MemoryService):
        self.client = silicon_expert_client
        self.memory = memory_service

    async def parse_components(self, raw_data: Dict[str, Any]) -> List[ComponentData]:
        """Parse raw component data into structured format."""
        components = raw_data.get('components', [])

        parsed_components = []
        for comp_data in components:
            component = ComponentData(
                name=comp_data.get('name', ''),
                part_number=comp_data.get('part_number'),
                manufacturer=comp_data.get('manufacturer'),
                description=comp_data.get('description', ''),
                value=comp_data.get('value'),
                designator=comp_data.get('designator'),
                confidence=float(comp_data.get('confidence', 0.8))
            )
            parsed_components.append(component)

        return parsed_components

    async def search_and_enhance(self, components: List[Dict[str, Any]]) -> List[ComponentData]:
        """Search and enhance component data."""
        enhanced_components = []

        for comp_data in components:
            try:
                # Search Silicon Expert
                search_result = await self.client.search_component(comp_data)

                # Create enhanced component
                component = ComponentData(
                    name=comp_data.get('name', ''),
                    part_number=search_result.get('part_number') or comp_data.get('part_number'),
                    manufacturer=search_result.get('manufacturer') or comp_data.get('manufacturer'),
                    description=search_result.get('description') or comp_data.get('description'),
                    value=comp_data.get('value'),
                    designator=comp_data.get('designator'),
                    confidence=search_result.get('confidence', comp_data.get('confidence', 0.5)),
                    metadata={
                        'enhanced': True,
                        'original_data': comp_data,
                        'search_result': search_result
                    }
                )

                enhanced_components.append(component)

            except Exception as e:
                # Fallback to original data
                component = ComponentData(
                    name=comp_data.get('name', ''),
                    part_number=comp_data.get('part_number'),
                    manufacturer=comp_data.get('manufacturer'),
                    description=comp_data.get('description'),
                    value=comp_data.get('value'),
                    designator=comp_data.get('designator'),
                    confidence=comp_data.get('confidence', 0.3),
                    metadata={'enhanced': False, 'error': str(e)}
                )
                enhanced_components.append(component)

        return enhanced_components
