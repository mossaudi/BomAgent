from typing import List, Dict, Any
from urllib.parse import urlparse

from src.core.models import ComponentData


def is_valid_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([
            result.scheme in ['http', 'https'],
            result.netloc,
            result.path
        ])
    except:
        return False

async def handle_api_call(client_method, *args, **kwargs):
    """Handle API call with common error handling"""
    try:
        return await client_method(*args, **kwargs)
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "data": None
        }


def convert_components_to_api_format(components: List[ComponentData]) -> List[Dict[str, Any]]:
    """Convert ComponentData to API format"""
    parts_data = []
    for comp in components:
        part = {
            "part_number": comp.part_number or comp.designator or "Unknown",
            "manufacturer": comp.manufacturer or "Unknown",
            "description": f"{comp.name} - {comp.description or comp.value or ''}".strip(" - "),
            "quantity": str(comp.quantity),
            "designator": comp.designator or ""
        }
        parts_data.append(part)
    return parts_data