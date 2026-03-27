from llm_parser import LLMParser


class IntentManager:
    def __init__(self):
        self.intent = {}
        self.parser = LLMParser()
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