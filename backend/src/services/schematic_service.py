# src/services/schematic_service.py
"""Simplified schematic analysis service with no timeouts."""

import json
import re
import asyncio
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class SchematicService:
    """Simplified schematic analysis service."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert electronics engineer analyzing circuit schematics. 
            Your task is to extract ALL visible components from the schematic image with maximum detail and accuracy.

            ANALYSIS REQUIREMENTS:
            - Identify EVERY visible component (resistors, capacitors, ICs, transistors, diodes, inductors, crystals, connectors, etc.)
            - Extract component values, reference designators, and part numbers where visible
            - Include component orientations and connections if relevant
            - Look for small components that might be easily missed
            - Identify integrated circuits by their package types and pin counts
            - Note any text labels, part numbers, or manufacturer markings
            - Include passive components even if values aren't clearly visible

            COMPONENT CATEGORIES TO IDENTIFY:
            1. Resistors (R1, R2, etc.) - extract values in ohms/kohms/mohms
            2. Capacitors (C1, C2, etc.) - extract values in pF/nF/uF/mF
            3. Inductors/Coils (L1, L2, etc.) - extract values in nH/uH/mH/H
            4. Integrated Circuits (U1, U2, IC1, etc.) - note package type, pin count
            5. Transistors (Q1, Q2, T1, etc.) - identify type (NPN/PNP/MOSFET/etc.)
            6. Diodes (D1, D2, etc.) - identify type (standard/LED/Zener/Schottky/etc.)
            7. Crystals/Oscillators (Y1, X1, etc.) - extract frequency
            8. Connectors (J1, P1, CN1, etc.) - note pin count and type
            9. Switches/Buttons (SW1, S1, etc.)
            10. Test points, jumpers, and other components

            RESPONSE FORMAT:
            Return ONLY a valid JSON object with this exact structure:
            {
              "components": [
                {
                  "name": "Component Type (e.g., Resistor, Capacitor, IC)",
                  "part_number": "Part number if visible (or null)",
                  "manufacturer": "Manufacturer if visible (or null)",
                  "description": "Detailed description including package type for ICs",
                  "value": "Component value with units (e.g., 10k, 100uF, 16MHz)",
                  "designator": "Reference designator (e.g., R1, C2, U1)",
                  "confidence": 0.9,
                  "category": "resistor|capacitor|inductor|ic|transistor|diode|crystal|connector|switch|other",
                  "package": "Package type for ICs (e.g., DIP-8, SOIC-16, QFN-32)",
                  "pins": "Pin count for ICs and connectors",
                  "voltage_rating": "Voltage rating if visible",
                  "tolerance": "Tolerance if visible (e.g., 5%, 1%)",
                  "power_rating": "Power rating if visible (e.g., 0.25W, 1W)"
                }
              ],
              "total_components": 0,
              "analysis_notes": "Any additional observations about the circuit"
            }

            QUALITY REQUIREMENTS:
            - Set confidence scores based on clarity: 0.95+ for clearly visible, 0.8+ for mostly clear, 0.6+ for partially visible
            - Include components even if you can only partially identify them
            - For unclear values, provide best estimate with lower confidence
            - Group similar components but list each instance separately

            CRITICAL: Return ONLY the JSON object, no additional text, explanations, or formatting."""),
            ("user", "Analyze this electronic schematic image and extract ALL visible components: {image_url}")
        ])

    async def analyze(self, image_url: str) -> Dict[str, Any]:
        """Analyze schematic with NO timeout limitations"""
        try:
            print(f"🔍 Starting schematic analysis for: {image_url}")

            # Format the prompt with image URL
            messages = self.prompt.format_messages(image_url=image_url)

            # Call LLM with NO timeout and large output limit
            print("🤖 Calling Gemini for analysis...")
            response = await self.llm.ainvoke(
                messages,
                temperature=0.1,  # Low temperature for consistent extraction
                max_output_tokens=50000,  # Increased output limit
                # NO timeout parameter - let it take as long as needed
            )

            content = response.content.strip()
            print(f"📝 Received response ({len(content)} characters)")

            # Extract JSON from response with improved regex
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

                    # Enhance component data
                    for i, comp in enumerate(result['components']):
                        # Ensure required fields exist
                        comp.setdefault('name', f'Unknown Component {i + 1}')
                        comp.setdefault('part_number', None)
                        comp.setdefault('manufacturer', None)
                        comp.setdefault('description', '')
                        comp.setdefault('value', None)
                        comp.setdefault('designator', f'COMP{i + 1}')
                        comp.setdefault('confidence', 0.5)
                        comp.setdefault('category', 'other')

                        # Ensure confidence is a float between 0 and 1
                        try:
                            comp['confidence'] = max(0.0, min(1.0, float(comp['confidence'])))
                        except (ValueError, TypeError):
                            comp['confidence'] = 0.5

                    print(f"🎯 Analysis complete: {len(result['components'])} components found")
                    return result

                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {str(e)}")
                    raise ValueError(f"Invalid JSON in LLM response: {str(e)}\nContent: {json_str[:500]}...")
            else:
                print(f"❌ No valid JSON found in response")
                raise ValueError(f"No valid JSON found in LLM response. Content: {content[:500]}...")

        except Exception as e:
            print(f"❌ Schematic analysis failed: {str(e)}")
            raise Exception(f"Schematic analysis failed: {str(e)}")

    async def analyze_with_retry(self, image_url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Analyze schematic with retry logic and NO timeouts"""
        last_error = None

        for attempt in range(max_retries):
            try:
                print(f"🔄 Analysis attempt {attempt + 1}/{max_retries}")
                result = await self.analyze(image_url)

                # Validate we got a reasonable number of components
                component_count = len(result.get('components', []))

                if component_count == 0:
                    print("⚠️ No components found in analysis")
                    if attempt < max_retries - 1:
                        print("⏳ Retrying analysis...")
                        await asyncio.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        # Return empty result on final attempt
                        return {
                            "components": [],
                            "total_components": 0,
                            "analysis_notes": "No components found in the schematic image"
                        }

                print(f"✅ Schematic analysis successful on attempt {attempt + 1}: {component_count} components found")
                return result

            except Exception as e:
                last_error = e
                print(f"❌ Analysis attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    # Wait before retry with exponential backoff (but no timeout)
                    wait_time = 2 ** attempt
                    print(f"⏳ Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

        # If all retries failed, return a minimal result with error info
        print(f"❌ All {max_retries} analysis attempts failed")
        return {
            "components": [],
            "total_components": 0,
            "analysis_notes": f"Analysis failed after {max_retries} attempts. Last error: {str(last_error)}",
            "error": str(last_error)
        }