# src/services/schematic_service_fixed.py
"""Fixed schematic analysis service with proper vision LLM integration."""

import asyncio
import json
import re
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class SchematicService:
    """Fixed schematic analysis service with proper vision integration."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    async def analyze(self, image_url: str) -> Dict[str, Any]:
        """Analyze schematic with proper vision LLM usage"""
        try:
            print(f"🔍 Starting schematic analysis for: {image_url}")

            # Create the system prompt
            system_content = """You are an expert electronics engineer analyzing circuit schematics. 
            Your task is to extract ALL visible components from the schematic image, focusing ONLY on information that can actually be seen in the image.

            ANALYSIS REQUIREMENTS:
            - Identify EVERY visible component (resistors, capacitors, ICs, transistors, diodes, inductors, crystals, connectors, etc.)
            - Extract ONLY the information that is clearly visible in the schematic
            - Focus on component values that are actually readable
            - Don't guess or assume information that isn't clearly shown
            - Include component descriptions only when helpful context is visible

            COMPONENT CATEGORIES TO IDENTIFY:
            1. Resistors (R1, R2, etc.) - extract visible values (1k, 10k, 470, etc.)
            2. Capacitors (C1, C2, etc.) - extract visible values (0.1uF, 10uF, 100pF, etc.)
            3. Inductors/Coils (L1, L2, etc.) - extract visible values if shown
            4. Integrated Circuits (U1, U2, IC1, etc.) - extract part numbers if visible
            5. Transistors (Q1, Q2, T1, etc.) - note type only if clearly marked
            6. Diodes (D1, D2, etc.) - identify type only if clearly shown (LED colors, etc.)
            7. Crystals/Oscillators (Y1, X1, etc.) - extract frequency if visible
            8. Connectors (J1, P1, CN1, etc.) - count pins if clearly countable
            9. Switches/Buttons (SW1, S1, etc.)
            10. Other components (fuses, test points, etc.)

            RESPONSE FORMAT - ONLY include properties that are actually visible:
            {
              "components": [
                {
                  "name": "Component Type (e.g., Resistor, Capacitor, IC)",
                  "designator": "Reference designator (e.g., R1, C2, U1)",
                  "value": "Component value with units ONLY if visible (e.g., 10k, 100uF, 16MHz)",
                  "part_number": "Part number ONLY if clearly readable",
                  "description": "meaningful description",
                  "confidence": 0.9,
                  "category": "resistor|capacitor|inductor|ic|transistor|diode|crystal|connector|switch|fuse|other"
                }
              ],
              "total_components": 0,
              "analysis_notes": "Brief observations about the circuit"
            }

            IMPORTANT RULES:
            - DO NOT include properties that are not visible (no null values)
            - DO NOT guess manufacturer, package type, voltage rating, tolerance, or power rating unless clearly shown
            - DO NOT include pins count unless you can clearly count them
            - Focus on what's actually readable in the image
            - Set confidence scores realistically: 0.95+ for crystal clear, 0.85+ for clearly readable, 0.7+ for mostly clear

            CRITICAL: Return ONLY the JSON object with visible properties, no additional text."""

            # Create message with image
            message_content = [
                {
                    "type": "text",
                    "text": f"{system_content}\n\nAnalyze this electronic schematic image and extract ALL visible components with their clearly readable properties:"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]

            message = HumanMessage(content=message_content)
            print("🤖 Calling Gemini Vision for optimized analysis...")

            # Call LLM with optimized parameters
            response = await self.llm.ainvoke(
                [message]
            )

            content = response.content.strip()
            print(f"📝 Received response ({len(content)} characters)")

            # Enhanced JSON extraction with better error handling
            json_match = re.search(r'\{.*\}', content, re.DOTALL | re.MULTILINE)

            if json_match:
                json_str = json_match.group()
                try:
                    result = json.loads(json_str)
                    print(f"✅ Successfully parsed JSON response")

                    # Validate and enhance the result
                    if 'components' not in result:
                        result['components'] = []

                    # Update total count
                    result['total_components'] = len(result['components'])

                    # Enhance component data with proper defaults
                    for i, comp in enumerate(result['components']):
                        # Ensure required fields exist with meaningful defaults
                        comp.setdefault('name', f'Unknown Component {i + 1}')
                        comp.setdefault('part_number', None)
                        comp.setdefault('manufacturer', None)
                        comp.setdefault('description', '')
                        comp.setdefault('value', None)
                        comp.setdefault('designator', f'COMP{i + 1}')
                        comp.setdefault('confidence', 0.5)
                        comp.setdefault('category', 'other')

                        # New fields for better component data
                        comp.setdefault('package', None)
                        comp.setdefault('pins', None)
                        comp.setdefault('voltage_rating', None)
                        comp.setdefault('tolerance', None)
                        comp.setdefault('power_rating', None)

                        # Ensure confidence is a float between 0 and 1
                        try:
                            comp['confidence'] = max(0.0, min(1.0, float(comp['confidence'])))
                        except (ValueError, TypeError):
                            comp['confidence'] = 0.5

                        # Add metadata about extraction quality
                        comp['extraction_quality'] = (
                            'high' if comp['confidence'] > 0.8 else
                            'medium' if comp['confidence'] > 0.5 else
                            'low'
                        )

                    # Add analysis metadata
                    result['analysis_metadata'] = {
                        'image_url': image_url,
                        'analysis_timestamp': asyncio.get_event_loop().time(),
                        'llm_model': 'gemini-2.0-flash',
                        'extraction_method': 'vision_llm'
                    }

                    print(f"🎯 Analysis complete: {len(result['components'])} components found")

                    # Log component summary for debugging
                    if result['components']:
                        print("📦 Found components:")
                        for i, comp in enumerate(result['components'][:5], 1):
                            value_str = f" ({comp['value']})" if comp.get('value') else ""
                            part_str = f" [{comp['part_number']}]" if comp.get('part_number') else ""
                            print(
                                f"  {i}. {comp['designator']}: {comp['name']}{value_str}{part_str} - {comp['confidence']:.2f}")
                        if len(result['components']) > 5:
                            print(f"  ... and {len(result['components']) - 5} more")

                    return result

                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {str(e)}")
                    print(f"Raw content preview: {json_str[:200]}...")
                    raise ValueError(f"Invalid JSON in LLM response: {str(e)}")
            else:
                print(f"❌ No valid JSON found in response")
                print(f"Raw content preview: {content[:500]}...")
                raise ValueError("No valid JSON or component patterns found in LLM response")

        except Exception as e:
            print(f"❌ Schematic analysis failed: {str(e)}")
            # Return a more informative error response
            return {
                'components': [],
                'total_components': 0,
                'analysis_notes': f"Analysis failed: {str(e)}",
                'error': str(e),
                'extraction_method': 'failed'
            }

    async def analyze_with_retry(self, image_url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Enhanced retry logic with exponential backoff"""
        last_error = None

        for attempt in range(max_retries):
            try:
                print(f"🔄 Analysis attempt {attempt + 1}/{max_retries}")
                result = await self.analyze(image_url)

                # Check if we got meaningful results
                component_count = len(result.get('components', []))

                if component_count == 0 and result.get('error'):
                    # If we have an error and no components, retry
                    print(f"⚠️ Analysis failed with error: {result.get('error')}")
                    last_error = result.get('error')

                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                        print(f"⏳ Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue

                # Success case or final attempt
                if component_count > 0:
                    print(f"✅ Analysis successful on attempt {attempt + 1}: {component_count} components found")
                else:
                    print(f"⚠️ Analysis complete but no components found on attempt {attempt + 1}")

                return result

            except Exception as e:
                last_error = str(e)
                print(f"❌ Analysis attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 10)
                    print(f"⏳ Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

        # All retries failed
        print(f"❌ All {max_retries} analysis attempts failed")
        return {
            "components": [],
            "total_components": 0,
            "analysis_notes": f"Analysis failed after {max_retries} attempts. Last error: {str(last_error)}",
            "error": str(last_error),
            "extraction_method": "failed_all_retries"
        }