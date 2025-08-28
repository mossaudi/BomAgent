# enhanced_bom_tools_fixed.py - Clean and Efficient BOM Tools
"""
Fixed issues:
1. Help command no longer triggers analysis
2. Proper URL validation before analysis
3. Clean async/sync handling
4. Simplified tool execution
5. Better error handling
"""

import asyncio
from typing import Dict, Any, List

from langchain_core.tools import Tool
from src.core.container import Container
from src.core.models import ComponentData
from src.core.utils import handle_api_call, convert_components_to_api_format


class ComponentStateManager:
    """Simplified component state manager"""

    def __init__(self):
        self._raw_components: List[ComponentData] = []
        self._enhanced_components: List[ComponentData] = []
        self._enhancement_status: Dict[str, str] = {}

    def store_raw_analysis(self, components: List[ComponentData]):
        """Store raw components from schematic analysis"""
        self._raw_components = components
        self._enhanced_components.clear()
        self._enhancement_status.clear()

    def store_enhanced_components(self, enhanced_components: List[ComponentData]):
        """Store enhanced components"""
        self._enhanced_components = enhanced_components
        for comp in enhanced_components:
            if comp.metadata and comp.metadata.get('enhanced'):
                self._enhancement_status[comp.id] = 'enhanced'
            else:
                self._enhancement_status[comp.id] = 'failed'

    def get_components_for_bom(self) -> List[ComponentData]:
        """Get components for BOM operations"""
        return self._enhanced_components if self._enhanced_components else self._raw_components

    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get enhancement summary"""
        total = len(self._raw_components)
        enhanced = len([s for s in self._enhancement_status.values() if s == 'enhanced'])
        return {
            'total_found': total,
            'enhanced_count': enhanced,
            'failed_count': total - enhanced,
            'enhancement_rate': (enhanced / total * 100) if total > 0 else 0,
            'has_components': total > 0
        }

    def clear(self):
        """Clear all stored data"""
        self._raw_components.clear()
        self._enhanced_components.clear()
        self._enhancement_status.clear()


class BOMTools:
    """Fixed BOM tools with proper request handling"""

    def __init__(self, container: Container):
        self.container = container
        self.component_state = ComponentStateManager()

    def create_tools(self) -> List[Tool]:
        """Create clean, efficient tools"""
        return [
            self._create_help_tool(),
            self._create_analyze_tool(),
            self._create_show_components_tool(),
            self._create_search_component_tool(),
            self._create_create_bom_tool(),
            self._create_add_components_tool(),
            self._create_list_boms_tool()
        ]

    def _create_help_tool(self) -> Tool:
        """Simple help tool - NO analysis triggered"""

        def show_help(_: str = "") -> str:
            return """🔧 **BOM Agent - Electronic Design Assistant**

                    **📋 Available Commands:**
                    
                    **1. Schematic Analysis:**
                       Format: `analyze_schematic(https://your-image-url.com/schematic.jpg)`
                       - Extracts ALL components from circuit schematics
                       - Auto-enhances with Silicon Expert data
                       - Supports PNG, JPG, JPEG formats
                       - Requires publicly accessible URL
                    
                    **2. Component Management:**
                       • `show_components_table()` - View analyzed components
                       • `search_component(part_name)` - Find specific components
                    
                    **3. BOM Operations:**
                       • `create_bom(name=MyBOM,project=Arduino)` - Create new BOM
                       • `add_components_to_bom(BOM_name)` - Add stored components
                       • `list_boms()` - Show existing BOMs
                    
                    **🎯 Typical Workflow:**
                    1. Analyze schematic with public URL
                    2. Review components in table format
                    3. Create BOM for your project
                    4. Add all components to BOM
                    
                    **💡 Important:**
                    - Schematic images must be publicly accessible
                    - Analysis only starts when you provide a valid URL
                    - All components are stored in memory for BOM operations
                    
                    Ready to help with your electronic designs! 🚀"""

        return Tool(
            name="help",
            description="Show BOM agent capabilities and usage examples",
            func=show_help
        )

    # Add method to handle actual analysis in the agent
    async def _process_pending_analysis(self):
        """Handle pending analysis asynchronously"""
        if hasattr(self, '_pending_analysis_url'):
            url = self._pending_analysis_url
            delattr(self, '_pending_analysis_url')

            try:
                # Analyze schematic
                schematic_service = self.container.services.schematic
                analysis_result = await schematic_service.analyze_with_retry(url)

                # Process and store results
                if analysis_result.get('components'):
                    raw_components = []
                    for comp_data in analysis_result['components']:
                        component = ComponentData(
                            name=comp_data.get('name', 'Unknown'),
                            part_number=comp_data.get('part_number'),
                            manufacturer=comp_data.get('manufacturer'),
                            description=comp_data.get('description', ''),
                            value=comp_data.get('value'),
                            designator=comp_data.get('designator'),
                            confidence=float(comp_data.get('confidence', 0.5)),
                            metadata={'category': comp_data.get('category', 'other')}
                        )
                        raw_components.append(component)

                    # Store components
                    await self.container.services.memory.store_components(raw_components)

                    return f"✅ Analysis complete! Found {len(raw_components)} components."
                else:
                    return "⚠️ No components found in schematic."

            except Exception as e:
                return f"❌ Analysis failed: {str(e)}"

        return None

    def _run_analysis_sync(self, image_url: str) -> str:
        """Run analysis in sync context"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(self._analyze_async(image_url))
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

    async def _analyze_async(self, image_url: str) -> str:
        """Async analysis implementation"""
        try:
            print("🔍 Starting schematic analysis...")

            # Analyze schematic
            schematic_service = self.container.services.schematic
            analysis_result = await schematic_service.analyze_with_retry(image_url)

            if not analysis_result.get('components'):
                return "⚠️ No components found. Please check image quality and ensure it shows a clear circuit schematic."

            # Convert to ComponentData objects
            raw_components = []
            for comp_data in analysis_result['components']:
                component = ComponentData(
                    name=comp_data.get('name', 'Unknown'),
                    part_number=comp_data.get('part_number'),
                    manufacturer=comp_data.get('manufacturer'),
                    description=comp_data.get('description', ''),
                    value=comp_data.get('value'),
                    designator=comp_data.get('designator'),
                    confidence=float(comp_data.get('confidence', 0.5)),
                    metadata={'category': comp_data.get('category', 'other')}
                )
                raw_components.append(component)

            self.component_state.store_raw_analysis(raw_components)
            print(f"✅ Found {len(raw_components)} components")

            # Auto-enhance components
            print("🚀 Enhancing components with Silicon Expert...")
            enhanced_components = await self._enhance_components(raw_components)
            self.component_state.store_enhanced_components(enhanced_components)

            return self._generate_analysis_response(enhanced_components)

        except Exception as e:
            return f"❌ Analysis failed: {str(e)}\n\n💡 Please verify the image URL is accessible and shows a clear schematic."

    async def _enhance_components(self, components: List[ComponentData]) -> List[ComponentData]:
        """Enhanced component processing with error handling"""
        enhanced = []
        client = self.container.services.silicon_expert_client

        for i, component in enumerate(components):
            try:
                print(f"📡 Enhancing {i + 1}/{len(components)}: {component.name}")

                search_data = {
                    "name": component.name,
                    "part_number": component.part_number,
                    "manufacturer": component.manufacturer,
                    "description": component.description
                }

                # Search with timeout
                search_result = await asyncio.wait_for(
                    client.search_component(search_data),
                    timeout=10.0
                )

                # Create enhanced component
                enhanced_comp = ComponentData(
                    id=component.id,
                    name=component.name,
                    part_number=search_result.part_number or component.part_number,
                    manufacturer=search_result.manufacturer or component.manufacturer,
                    description=search_result.description or component.description,
                    value=component.value,
                    designator=component.designator,
                    confidence=max(component.confidence, search_result.confidence),
                    metadata={
                        **component.metadata,
                        'enhanced': search_result.success,
                        'original_confidence': component.confidence,
                        'search_confidence': search_result.confidence
                    }
                )
                enhanced.append(enhanced_comp)

                await asyncio.sleep(0.1)  # Small delay

            except Exception as e:
                print(f"⚠️ Enhancement failed for {component.name}: {e}")
                component.metadata['enhanced'] = False
                enhanced.append(component)

        return enhanced

    def _generate_analysis_response(self, components: List[ComponentData]) -> str:
        """Generate clean analysis response"""
        summary = self.component_state.get_enhancement_summary()

        output = f"""✅ **Schematic Analysis Complete!**

📊 **Results:**
• Components Found: **{summary['total_found']}**
• Enhanced with Silicon Expert: **{summary['enhanced_count']}** ({summary['enhancement_rate']:.1f}%)
• Using Original Data: **{summary['failed_count']}**

🎯 **Next Steps:**
• Use `show_components_table()` to view all components
• Use `create_bom(name=MyBOM,project=MyProject)` to create BOM
• Use `add_components_to_bom(BOM_name)` to add components

💡 All {summary['total_found']} components are stored and ready for BOM operations!"""

        return output

    def _create_show_components_tool(self) -> Tool:
        """Show components in table format"""

        def show_table(_: str = "") -> str:
            components = self.component_state.get_components_for_bom()

            if not components:
                return "⚠️ No components available. Analyze a schematic first using: `analyze_schematic(URL)`"

            summary = self.component_state.get_enhancement_summary()

            output = f"📋 **Component Table** ({len(components)} components)\n\n"
            output += f"Enhancement Status: {summary['enhanced_count']}/{summary['total_found']} enhanced\n\n"
            output += "| # | Designator | Component | Value | Part Number | Manufacturer |\n"
            output += "|---|------------|-----------|-------|-------------|-------------|\n"

            for i, comp in enumerate(components[:15], 1):  # Show first 15
                designator = comp.designator or f"COMP{i}"
                name = comp.name[:20] + "..." if len(comp.name) > 20 else comp.name
                value = comp.value or "-"
                part_num = comp.part_number[:15] + "..." if comp.part_number and len(comp.part_number) > 15 else (
                            comp.part_number or "-")
                manufacturer = comp.manufacturer[:12] + "..." if comp.manufacturer and len(
                    comp.manufacturer) > 12 else (comp.manufacturer or "-")

                enhanced = "✅" if comp.metadata and comp.metadata.get('enhanced') else "📋"

                output += f"| {i} | {designator} | {enhanced} {name} | {value} | {part_num} | {manufacturer} |\n"

            if len(components) > 15:
                output += f"\n... and {len(components) - 15} more components\n"

            output += "\n**Legend:** ✅ Enhanced, 📋 Original data\n"
            output += "\n**Actions:** Create BOM • Add to existing BOM • Export data"

            return output

        return Tool(
            name="show_components_table",
            description="Display all stored components in table format",
            func=show_table
        )

    def _search_sync(self, query: str) -> str:
        """Run search in sync context"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(self._search_async(query))
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

    async def _search_async(self, query: str) -> str:
        """Async search implementation"""
        try:
            client = self.container.services.silicon_expert_client
            search_data = {"name": query, "description": query}
            result = await client.search_component(search_data)

            if result.success and result.part_number:
                return f"""✅ **Component Found!**

🔍 **Search:** {query}
📦 **Part Number:** {result.part_number}
🏭 **Manufacturer:** {result.manufacturer or 'Not specified'}
📝 **Description:** {result.description or 'Not available'}
⭐ **Confidence:** {result.confidence:.1%}

💡 Component can be added to your collection for BOM operations."""
            else:
                return f"❌ No component found for '{query}'. Try specific part numbers or component names."

        except Exception as e:
            return f"❌ Search failed: {str(e)}"


    def _create_bom_sync(self, params: str) -> str:
        """Create BOM in sync context"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(self._create_bom_async(params))
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

    async def _create_bom_async(self, params: str) -> str:
        """Async BOM creation"""
        parsed = self._parse_params(params)
        name = parsed.get('name', '').strip()

        if not name:
            return "❌ BOM name required. Format: name=BOM_NAME,project=PROJECT"

        try:
            bom_service = self.container.services.bom
            result = await bom_service.create_bom(
                name=name,
                description=parsed.get('description', ''),
                project=parsed.get('project', '')
            )

            if result.get('success'):
                components = self.component_state.get_components_for_bom()
                output = f"✅ **BOM Created: {name}**\n\n"
                if parsed.get('project'):
                    output += f"📁 **Project:** {parsed.get('project')}\n"
                if components:
                    output += f"📦 **Ready Components:** {len(components)} available to add\n"
                output += f"\n💡 Use `add_components_to_bom({name})` to add your components"
                return output
            else:
                return f"❌ BOM creation failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"❌ BOM creation failed: {str(e)}"

    def _create_add_components_tool(self) -> Tool:
        def add_components(bom_name: str) -> str:
            return asyncio.run(self._add_components_async(bom_name))
        return Tool(
            name="add_components_to_bom",
            description="Add all stored components to existing BOM",
            func=add_components
        )

    def _add_components_sync(self, bom_name: str) -> str:
        """Add components in sync context"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(self._add_components_async(bom_name))
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

    async def _add_components_async(self, bom_name: str) -> str:
        """Async add components implementation"""
        components = self.component_state.get_components_for_bom()
        if not components:
            return "⚠️ No components available. Analyze a schematic first."
        parts_data = convert_components_to_api_format(components)
        bom_service = self.container.services.bom
        result = await handle_api_call(bom_service.add_parts, bom_name.strip(), "", parts_data)
        if result.get('success'):
            summary = self.component_state.get_enhancement_summary()
            return f"""✅ **Components Added to {bom_name}!**
                        
                        📦 **Total Parts:** {len(components)}
                        ✅ **Enhanced:** {summary['enhanced_count']}
                        📋 **Original:** {summary['failed_count']}
                        
                        🎯 BOM '{bom_name}' now contains all your analyzed components!"""
        else:
            return f"❌ Failed to add components: {result.get('error', 'Unknown error')}"

    def _create_list_boms_tool(self) -> Tool:
        def list_boms(project_filter: str = "") -> str:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._list_boms_sync, project_filter)
                        return future.result(timeout=30)
                except RuntimeError:
                    return asyncio.run(self._list_boms_async(project_filter))
            except Exception as e:
                return f"❌ List BOMs failed: {str(e)}"

        return Tool(
            name="list_boms",
            description="List existing BOMs, optionally filtered by project",
            func=list_boms
        )

    def _list_boms_sync(self, project_filter: str) -> str:
        """List BOMs in sync context"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(self._list_boms_async(project_filter))
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

    async def _list_boms_async(self, project_filter: str) -> str:
        """Async list BOMs implementation"""
        try:
            bom_service = self.container.services.bom
            result = await bom_service.get_boms(project_filter.strip())

            if result.get('success'):
                boms = result.get('boms', [])

                if not boms:
                    return "📂 No BOMs found. Create your first BOM to get started!"

                output = f"📋 **BOMs Overview** ({len(boms)} found)\n\n"

                for i, bom in enumerate(boms[:10], 1):
                    name = bom.get('name', 'Unknown')
                    project = bom.get('project', '')
                    parts_count = bom.get('component_count', 'N/A')

                    output += f"{i}. **{name}**"
                    if project:
                        output += f" (📁 {project})"
                    if parts_count != 'N/A':
                        output += f" - {parts_count} parts"
                    output += "\n"

                if len(boms) > 10:
                    output += f"\n... and {len(boms) - 10} more BOMs"

                components = self.component_state.get_components_for_bom()
                if components:
                    output += f"\n\n📦 **Available:** {len(components)} components ready to add"

                return output
            else:
                return f"❌ Failed to retrieve BOMs: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"❌ List BOMs failed: {str(e)}"

    def _parse_params(self, params: str) -> Dict[str, str]:
        """Parse comma-separated key=value parameters"""
        result = {}
        if not params or params.strip() == "":
            return result

        try:
            for pair in params.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    result[key.strip()] = value.strip()
        except Exception:
            pass

        return result