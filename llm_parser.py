import os
import json
from groq import Groq
from dotenv import load_dotenv

# Ensure environment variables are loaded if using standalone
load_dotenv()

class LLMParser:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
             print("⚠ Warning: GROQ_API_KEY not found in environment.")
        self.client = Groq(api_key=self.api_key)

    def parse(self, query):
        prompt = f"""
Convert this CCTV query into a structured JSON string. Focus on filtering specific COCO classes (Person=0, Car=2, Bike=3, Bus=5, Truck=7).

Query: "{query}"

Respond ONLY with valid JSON in this format:
{{
  "object": "object_name",
  "cls_id": [class_ids],
  "color": "color_description",
  "event": "event_type",
  "zone": "zone_name",
  "time": "time_info"
}}

Class Mapping: {{0: "Person", 2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}}
"""

        models = [
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "llama-3.1-8b-instant"
        ]

        for model in models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )

                content = response.choices[0].message.content.strip()
                
                # Strip potential markdown blocks
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                     content = content[3:-3].strip()

                return json.loads(content)

            except Exception as e:
                # Silently try next model if it's a model-specific failure
                pass

        return {}