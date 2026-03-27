import cv2
from detector import HumanDetector
from tracker import PersonTracker


def play_event(video_file, video_time, target_id):

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Could not open video.")
        return

    detector = HumanDetector()
    tracker = PersonTracker()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(video_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    print(f"🎯 Jumping to {video_time:.2f}s (Target ID: {target_id})")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_objects = tracker.update(frame, detections)

        for (x1, y1, x2, y2, track_id, cls_id) in tracked_objects:

            if track_id == target_id:
                color = (0, 0, 255)  # RED for target
                label = f"TARGET ID {track_id}"
            else:
                color = (0, 255, 0)
                label = f"ID {track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Event Playback", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()