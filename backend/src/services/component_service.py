# src/services/component_service.py
"""Modern component search and enhancement service."""
from datetime import datetime
from typing import List, Dict, Any

from src.clients.silicon_expert_client import SiliconExpertClient
from src.services.memory_service import MemoryService
from src.core.models import ComponentData


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
                # Search Silicon Expert - handle both ComponentData and dict inputs
                if isinstance(comp_data, ComponentData):
                    search_data = {
                        'name': comp_data.name,
                        'part_number': comp_data.part_number,
                        'manufacturer': comp_data.manufacturer,
                        'description': comp_data.description,
                        'value': comp_data.value
                    }
                    base_component = comp_data
                else:
                    search_data = comp_data
                    base_component = ComponentData(
                        name=comp_data.get('name', ''),
                        part_number=comp_data.get('part_number'),
                        manufacturer=comp_data.get('manufacturer'),
                        description=comp_data.get('description', ''),
                        value=comp_data.get('value'),
                        designator=comp_data.get('designator'),
                        confidence=float(comp_data.get('confidence', 0.5))
                    )

                search_result = await self.client.search_component(search_data)

                # Create enhanced component
                component = ComponentData(
                    id=base_component.id,
                    name=base_component.name,
                    part_number=search_result.part_number or base_component.part_number,
                    manufacturer=search_result.manufacturer or base_component.manufacturer,
                    description=search_result.description or base_component.description,
                    value=base_component.value,
                    designator=base_component.designator,
                    confidence=max(search_result.confidence, base_component.confidence),
                    enhanced=search_result.success,
                    category=base_component.category,
                    metadata={
                        'enhanced': search_result.success,
                        'original_data': search_data,
                        'search_result': search_result.__dict__ if search_result else None,
                        'enhancement_timestamp': str(datetime.now())
                    }
                )

                enhanced_components.append(component)

            except Exception as e:
                print(f"Enhancement failed for component {comp_data}: {e}")
                # Fallback to original data
                if isinstance(comp_data, ComponentData):
                    component = comp_data
                    component.metadata = component.metadata or {}
                    component.metadata.update({'enhanced': False, 'error': str(e)})
                else:
                    component = ComponentData(
                        name=comp_data.get('name', ''),
                        part_number=comp_data.get('part_number'),
                        manufacturer=comp_data.get('manufacturer'),
                        description=comp_data.get('description'),
                        value=comp_data.get('value'),
                        designator=comp_data.get('designator'),
                        confidence=float(comp_data.get('confidence', 0.3)),
                        enhanced=False,
                        metadata={'enhanced': False, 'error': str(e)}
                    )
                enhanced_components.append(component)

        return enhanced_components
