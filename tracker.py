import numpy as np
import supervision as sv


class PersonTracker:
    def __init__(self):
        # Initialize ByteTrack tracker
        self.tracker = sv.ByteTrack()

    def update(self, frame, detections):

        if len(detections) == 0:
            return []

        boxes = []
        confidences = []
        class_ids = []

        # Unpack detections
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(cls_id)

        # Convert to numpy arrays
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        class_ids = np.array(class_ids)

        # Create supervision detections object
        detections_sv = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )

        # Run tracker
        tracked = self.tracker.update_with_detections(detections_sv)

        results = []

        # Extract tracked objects
        for i in range(len(tracked.xyxy)):

            x1, y1, x2, y2 = tracked.xyxy[i]
            track_id = tracked.tracker_id[i]
            cls_id = tracked.class_id[i]

            results.append((
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                int(track_id),
                int(cls_id)
            ))

        return results