from llm_parser import LLMParser
import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class IntentManager:
    def __init__(self):
        self.intent = {}
        self.parser = LLMParser()
        self._groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.COCO_MAPPING = {
            "person": [0],
            "human": [0],
            "people": [0],
            "car": [2],
            "vehicle": [2, 5, 7],
            "auto": [2],
            "bike": [3],
            "cycle": [3],
            "motorcycle": [3],
            "bus": [5],
            "truck": [7]
        }

    def set_intent(self, query):
        self.intent = self.parser.parse(query)
        print("🧠 Parsed Intent:", self.intent)

    def match(self, cls_id):
        # 1. Check if LLM directly provided cls_id list
        target_ids = self.intent.get("cls_id")
        if isinstance(target_ids, list) and target_ids:
             return cls_id in target_ids
        
        # 2. Check if LLM provided it as a single integer
        if isinstance(target_ids, int):
             return cls_id == target_ids

        # 3. Fallback to object string mapping
        obj = self.intent.get("object", "").lower()
        if not obj:
            return True # Show all if no object intent specified
            
        for key, ids in self.COCO_MAPPING.items():
             if key in obj:
                  return cls_id in ids

        return True # Default to show if unsure

    def parse_search_query(self, query):
        """Parse a natural-language forensic search query into a structured filter dict."""
        system_prompt = """You are a CCTV search query parser. 
Convert the user query into a JSON filter object with these fields 
(omit fields not mentioned, never add extra text):
{
  "object_class": "person|car|vehicle|bike|bus|truck",
  "color_label":  "silver|red|black|white|gray|blue|green|orange|yellow|purple",
  "zone_name":    "Zone A|Entrance|Parking|...",
  "floor":        "1|2|G|...",
  "group_id":     "floor1_zone_a|entrance_cluster|parking_perimeter",
  "time_from":    "HH:MM or ISO timestamp",
  "time_to":      "HH:MM or ISO timestamp",
  "event_type":   "intrusion|suspicious_stay|detection"
}
Respond with ONLY valid JSON, no markdown, no explanation."""

        try:
            response = self._groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=300
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            filters = json.loads(content)

            # Resolve scope to cam_ids
            try:
                import camera_groups
                scope = filters.get("group_id") or filters.get("floor") or filters.get("zone_name")
                if scope:
                    cam_ids = camera_groups.resolve_scope(scope)
                    if cam_ids:
                        filters["cam_ids"] = cam_ids
            except Exception:
                pass

            return filters
        except Exception as e:
            print(f"⚠ parse_search_query error: {e}")
            return {}