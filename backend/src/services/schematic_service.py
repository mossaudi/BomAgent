# src/services/schematic_service.py
"""Modern schematic analysis service."""

import json
import re
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class SchematicService:
    """Service for schematic analysis using LLM."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the schematic image and extract component information.
            Return a JSON object with 'components' array. Each component should have:
            - name: component identifier
            - part_number: if visible
            - manufacturer: if visible  
            - description: component type/function
            - value: component value if applicable
            - designator: reference designator (R1, C2, etc.)
            - confidence: confidence score 0-1

            Return ONLY valid JSON, no additional text."""),
            ("user", "Analyze schematic at: {image_url}")
        ])

    async def analyze(self, image_url: str) -> Dict[str, Any]:
        """Analyze schematic and return component data."""
        try:
            messages = self.prompt.format_messages(image_url=image_url)
            response = await self.llm.ainvoke(messages)

            # Extract JSON from response
            content = response.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)

            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in LLM response")

        except Exception as e:
            raise Exception(f"Schematic analysis failed: {str(e)}")
