# tools.py - ENHANCED WITH SESSION MEMORY
"""Enhanced LangGraph tools with session-based memory management."""
import json
from typing import List, Dict, Any

from langchain.tools import tool

from backend.src.core.models import BOMTreeResult
from backend.src.services.memory import get_memory_manager, get_component_memory
from container import Container
from exceptions import AgentError

# Global container instance
_container: Container = None


def initialize_tools(container: Container):
    """Initializes the tools with the dependency container."""
    global _container
    _container = container


def get_tools(container: Container) -> List[Any]:
    """Get all available tools, ensuring they are initialized."""
    initialize_tools(container)
    return [
        analyze_schematic,
        search_component_data,
        create_empty_bom,
        get_boms,
        add_parts_to_bom,
        get_last_components,
        get_memory_status,  # New tool for memory management
        clear_memory,  # New tool for memory cleanup
    ]


@tool
def analyze_schematic(image_url: str, session_id: str = None) -> str:
    """🔍 ANALYZE SCHEMATIC: Extracts and enhances component data from a schematic URL.

    This tool runs a complete workflow:
    1. Analyzes the schematic image to identify components.
    2. Searches for detailed data for each component.
    3. Returns a formatted table with all findings.
    4. Stores results in session memory for future use.

    Args:
        image_url: The public URL of the schematic image to analyze.
        session_id: Optional session ID (if not provided, uses current session)

    Returns:
        A formatted table showing all components with their enhanced data.
    """
    if not _container:
        raise AgentError("Tools not initialized.")

    # Get memory services
    memory_manager = get_memory_manager()
    component_memory = get_component_memory()

    # Set session if provided
    if session_id:
        memory_manager.set_session(session_id)

    # Ensure we have a session
    if not memory_manager.get_session():
        session_id = memory_manager.create_session()
        print(f"📝 Created new session: {session_id}")

    # Get the search result from workflow service
    search_result = _container.workflow_service.run_schematic_analysis_workflow(image_url)

    # Store in session memory
    component_memory.store_analysis_result(search_result, image_url)

    # Store analysis history
    existing_history = memory_manager.retrieve('analysis_history') or []
    existing_history.append({
        'image_url': image_url,
        'timestamp': __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
        'component_count': len(search_result.components),
        'success_rate': search_result.success_rate,
        'session_id': memory_manager.get_session()
    })
    memory_manager.store('analysis_history', existing_history)

    # Format and return the table directly
    formatted_result = _container.formatter.format_search_result(search_result)

    return formatted_result


@tool
def get_last_components(session_id: str = None) -> str:
    """📋 GET LAST COMPONENTS: Retrieves the components from the most recent schematic analysis.

    This allows you to access previously analyzed components without re-running the analysis.
    Use this when you want to:
    - Create a BOM from previously analyzed components
    - Add components to an existing BOM
    - Review component details again

    Args:
        session_id: Optional session ID (if not provided, uses current session)

    Returns:
        Formatted table of the last analyzed components, or a message if no components are available.
    """
    # Get memory services
    memory_manager = get_memory_manager()
    component_memory = get_component_memory()

    # Set session if provided
    if session_id:
        memory_manager.set_session(session_id)

    # Check if we have components
    if not component_memory.has_components():
        current_session = memory_manager.get_session()
        return f"❌ No previous component analysis found in session '{current_session}'. Please analyze a schematic first using 'analyze_schematic'."

    search_result = component_memory.get_last_search_result()
    formatted_result = _container.formatter.format_search_result(search_result)

    return formatted_result


@tool
def get_memory_status(session_id: str = None) -> str:
    """🧠 GET MEMORY STATUS: Shows the current session's memory status and available data.

    Args:
        session_id: Optional session ID (if not provided, uses current session)

    Returns:
        Formatted summary of the session's memory contents.
    """
    memory_manager = get_memory_manager()
    component_memory = get_component_memory()

    # Set session if provided
    if session_id:
        memory_manager.set_session(session_id)

    current_session = memory_manager.get_session()
    if not current_session:
        return "❌ No active session. Create a session by analyzing a schematic or using other tools."

    # Get session summary
    session_summary = memory_manager.get_session_summary()
    component_summary = component_memory.get_component_summary()

    # Format output
    output = f"""
        🧠 SESSION MEMORY STATUS
        {'=' * 50}
        📝 Session ID: {current_session}
        📊 Total Memory Keys: {session_summary['total_keys']}
        🕒 Session Created: {__import__('time').strftime('%Y-%m-%d %H:%M:%S', __import__('time').localtime(session_summary['created_at']))}
        
        🔧 COMPONENT DATA:
        {'=' * 30}
        """

    if component_summary['has_components']:
        output += f"✅ Components Available: {component_summary['component_count']} components\n"
        output += f"📈 Success Rate: {component_summary.get('success_rate', 0):.1f}%\n"

        if 'component_names' in component_summary:
            output += f"🏷️  Component Names: {', '.join(component_summary['component_names'])}\n"
    else:
        output += "❌ No components in memory\n"

    # Show analysis history if available
    history = memory_manager.retrieve('analysis_history')
    if history:
        output += f"\n📈 ANALYSIS HISTORY:\n{'=' * 30}\n"
        for i, analysis in enumerate(history[-3:], 1):  # Show last 3
            output += f"{i}. {analysis['timestamp']} - {analysis['component_count']} components ({analysis['success_rate']:.1f}%)\n"

        if len(history) > 3:
            output += f"   ... and {len(history) - 3} more analyses\n"

    # Show available memory keys
    if session_summary['total_keys'] > 0:
        output += f"\n🗂️  MEMORY KEYS:\n{'=' * 30}\n"
        for key in session_summary['keys']:
            key_detail = session_summary['key_details'].get(key, {})
            timestamp = key_detail.get('timestamp', 0)
            time_str = __import__('time').strftime('%H:%M:%S', __import__('time').localtime(timestamp))
            output += f"• {key} (stored at {time_str})\n"

    return output


@tool
def clear_memory(session_id: str = None, confirm: bool = False) -> str:
    """🗑️ CLEAR MEMORY: Clears the current session's memory.

    Args:
        session_id: Optional session ID (if not provided, uses current session)
        confirm: Set to True to actually clear the memory (safety measure)

    Returns:
        Confirmation message.
    """
    memory_manager = get_memory_manager()

    # Set session if provided
    if session_id:
        memory_manager.set_session(session_id)

    current_session = memory_manager.get_session()
    if not current_session:
        return "❌ No active session to clear."

    if not confirm:
        return f"""
            ⚠️  CLEAR MEMORY CONFIRMATION REQUIRED
            {'=' * 40}
            This will clear all memory for session: {current_session}
            
            To confirm, call: clear_memory(confirm=True)
            
            This will remove:
            • Component analysis results
            • Analysis history  
            • All stored data for this session
            """

    # Clear the session
    memory_manager.clear_session(current_session)

    return f"✅ Memory cleared for session: {current_session}"


@tool
def search_component_data(components_json: str, session_id: str = None) -> str:
    """🔍 SEARCH COMPONENT DATA: Searches for data on a given list of components.

    Args:
        components_json: A JSON string representing a list of components to search.
        session_id: Optional session ID (if not provided, uses current session)

    Returns:
        A formatted table with search results.
    """
    if not _container:
        raise AgentError("Tools not initialized.")

    # Get memory services
    memory_manager = get_memory_manager()
    component_memory = get_component_memory()

    # Set session if provided
    if session_id:
        memory_manager.set_session(session_id)

    # Ensure we have a session
    if not memory_manager.get_session():
        memory_manager.create_session()

    # Delegate parsing and searching to the respective services
    components = _container.parsing_service.parse_and_convert_to_components(components_json)
    search_result = _container.analysis_service.search_component_data(components)

    # Store in session memory
    component_memory.store_analysis_result(search_result)

    # Return formatted table
    return _container.formatter.format_search_result(search_result)


@tool
def create_empty_bom(name: str, columns: str, description: str = "", parent_path: str = "", session_id: str = None) -> \
Dict[str, Any]:
    """📋 CREATE EMPTY BOM: Create new Bill of Materials structure for custom BOM projects.

    Args:
        name: New BOM's name (mandatory)
        columns: JSON array of column names (optional)
        description: New BOM's description (optional)
        parent_path: Saving path in pattern "project>subproject1>subproject2" (optional)
        session_id: Optional session ID (if not provided, uses current session)

    Returns:
        JSON response with operation status
    """
    if not _container:
        raise AgentError("Tools not initialized.")

    # Handle session
    memory_manager = get_memory_manager()
    if session_id:
        memory_manager.set_session(session_id)

    try:
        columns_list = json.loads(columns) if isinstance(columns, str) else columns
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for columns"}

    result = _container.bom_service.create_empty_bom(
        name=name, columns=columns_list, description=description, parent_path=parent_path
    )

    # Store BOM creation in memory for reference
    if memory_manager.get_session():
        bom_history = memory_manager.retrieve('bom_history') or []
        bom_history.append({
            'action': 'create_empty',
            'name': name,
            'timestamp': __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
            'success': result.get('status', {}).get('success') == 'TRUE'
        })
        memory_manager.store('bom_history', bom_history)

    return result


@tool
def get_boms(project_name: str = "", session_id: str = None, **kwargs) -> str:
    """📋 GET BOMS: Lists existing Bills of Materials in a hierarchical tree view."""
    if not _container:
        raise AgentError("Tools not initialized.")

    # Handle session
    memory_manager = get_memory_manager()
    if session_id:
        memory_manager.set_session(session_id)

    try:
        bom_result = _container.bom_service.get_boms(project_name=project_name, **kwargs)

        if not bom_result.get("success", False):
            return f"❌ Failed to retrieve BOMs"

        bom_tree = BOMTreeResult(**bom_result["bom_tree"])
        return _container.formatter.format_bom_tree(bom_tree)

    except Exception as e:
        return f"❌ Error retrieving BOMs: {str(e)}"


@tool
def add_parts_to_bom(name: str, parent_path: str, parts_json: str = "", session_id: str = None) -> Dict[str, Any]:
    """➕ ADD PARTS TO BOM: Adds components to an existing BOM.

    Args:
        name: BOM name
        parent_path: BOM parent path (project path)
        parts_json: JSON string of parts data (optional - will use last analyzed components if empty)
        session_id: Optional session ID (if not provided, uses current session)

    Returns:
        Operation result
    """
    if not _container:
        raise AgentError("Tools not initialized.")

    # Get memory services
    memory_manager = get_memory_manager()
    component_memory = get_component_memory()

    # Set session if provided
    if session_id:
        memory_manager.set_session(session_id)

    # If no parts_json provided, use last analyzed components
    if not parts_json.strip():
        if not component_memory.has_components():
            return {
                "error": "No parts data provided and no previous component analysis found. "
                         "Please analyze a schematic first or provide parts_json."
            }

        # Convert last analyzed components to BOM parts format
        components = component_memory.get_last_components()
        bom_parts = []

        for component in components:
            if hasattr(component, 'to_bom_part'):
                bom_part = component.to_bom_part()
            else:
                # Fallback for dict-based components
                bom_part = {
                    'mpn': component.get('effective_part_number') or component.get('part_number', 'Unknown'),
                    'manufacturer': component.get('effective_manufacturer') or component.get('manufacturer', 'Unknown'),
                    'description': component.get('effective_description') or component.get('description',
                                                                                           'No description'),
                    'quantity': component.get('quantity', '1'),
                    'uploadedcomments': 'Added from schematic analysis'
                }

                if component.get('designator'):
                    bom_part['designator'] = component['designator']

            bom_parts.append(bom_part)

        parts_json = json.dumps(bom_parts)
        print(f"✅ Using {len(bom_parts)} components from session memory")

    result = _container.bom_service.add_parts_to_bom(
        name=name, parent_path=parent_path, parts_data=parts_json
    )

    # Store BOM operation in memory for reference
    if memory_manager.get_session():
        bom_history = memory_manager.retrieve('bom_history') or []
        bom_history.append({
            'action': 'add_parts',
            'bom_name': name,
            'parts_count': len(json.loads(parts_json)) if parts_json else 0,
            'timestamp': __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
            'success': result.get('Status', {}).get('Success') == 'true'
        })
        memory_manager.store('bom_history', bom_history)

    return result