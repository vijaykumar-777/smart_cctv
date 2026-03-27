import os
import cv2
import threading
import time
from collections import deque
from datetime import datetime


class ClipRecorder:
    def __init__(self, buffer_seconds=30, post_event_seconds=10, base_fps=15):
        self._buffers = {}          # cam_id -> deque of frames
        self._locks = {}            # cam_id -> threading.Lock
        self._global_lock = threading.Lock()
        self.buffer_seconds = buffer_seconds
        self.post_event_seconds = post_event_seconds
        self.base_fps = base_fps
        self.max_frames = buffer_seconds * base_fps
        os.makedirs("clips", exist_ok=True)
        os.makedirs("thumbs", exist_ok=True)

    def _get_lock(self, cam_id):
        with self._global_lock:
            if cam_id not in self._locks:
                self._locks[cam_id] = threading.Lock()
                self._buffers[cam_id] = deque(maxlen=self.max_frames)
            return self._locks[cam_id]

    def push_frame(self, cam_id, frame):
        lock = self._get_lock(cam_id)
        with lock:
            self._buffers[cam_id].append(frame.copy())

    def trigger_clip(self, cam_id, event_type, track_id, zone_name=None):
        """Non-blocking: spawns a thread to write the clip."""
        t = threading.Thread(
            target=self._write_clip,
            args=(cam_id, event_type, track_id, zone_name),
            daemon=True
        )
        t.start()

    def _write_clip(self, cam_id, event_type, track_id, zone_name):
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            ts_str = now.strftime("%H%M%S")
            cam_str = str(cam_id)

            clip_dir = os.path.join("clips", cam_str, date_str)
            thumb_dir = os.path.join("thumbs", cam_str, date_str)
            os.makedirs(clip_dir, exist_ok=True)
            os.makedirs(thumb_dir, exist_ok=True)

            filename = f"{ts_str}_{event_type}.mp4"
            clip_path = os.path.join(clip_dir, filename)
            thumb_path = os.path.join(thumb_dir, f"{ts_str}.jpg")

            # 1. Flush the buffer
            lock = self._get_lock(cam_id)
            with lock:
                pre_frames = list(self._buffers[cam_id])

            if not pre_frames:
                return

            h, w = pre_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(clip_path, fourcc, self.base_fps, (w, h))

            for f in pre_frames:
                writer.write(f)

            # 2. Continue recording post-event window
            end_time = time.time() + self.post_event_seconds
            while time.time() < end_time:
                with lock:
                    if self._buffers[cam_id]:
                        writer.write(self._buffers[cam_id][-1].copy())
                time.sleep(1.0 / self.base_fps)

            writer.release()

            # 3. Extract thumbnail (middle frame)
            all_frames = pre_frames
            mid = len(all_frames) // 2
            if 0 <= mid < len(all_frames):
                cv2.imwrite(thumb_path, all_frames[mid])

            print(f"📹 Clip saved: {clip_path} | Thumb: {thumb_path}")

            # 4. Update detection_index if available
            try:
                from db import _update_clip_paths
                _update_clip_paths(cam_str, track_id, clip_path, thumb_path)
            except Exception:
                pass

        except Exception as e:
            print(f"❌ ClipRecorder error: {e}")

    def get_clip_path(self, cam_id, timestamp):
        """Returns clip path for a given cam_id and timestamp string, or None."""
        cam_str = str(cam_id)
        if isinstance(timestamp, str) and len(timestamp) >= 10:
            date_str = timestamp[:10]
            time_str = timestamp[11:19].replace(":", "") if len(timestamp) >= 19 else ""
            clip_dir = os.path.join("clips", cam_str, date_str)
            if os.path.isdir(clip_dir):
                for f in os.listdir(clip_dir):
                    if f.startswith(time_str[:4]):
                        return os.path.join(clip_dir, f)
        return None
