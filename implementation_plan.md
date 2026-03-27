# Cross-Camera ReID & Performance Fix

## Problem
1. **Still laggy** — YOLO runs on full 1920×1080 frames
2. **IDs don't persist across cameras** — each camera has its own independent ByteTrack tracker
3. **cam-01 (view-HC3.mp4) should be removed** — focus on the two hall cameras

## Proposed Changes

### Performance — Frame Downscaling

#### [MODIFY] [app.py](file:///c:/Users/haris/Projects/smart_cctv/app.py)

- **Downscale** frames to 640px width before YOLO inference (3x fewer pixels, ~3x faster)
- Map detection boxes back to original resolution for drawing
- Keep display at original resolution for quality
- Remove `cam-01` from the `CAMERAS` dict

---

### Cross-Camera Person Re-Identification

#### [NEW] [global_tracker.py](file:///c:/Users/haris/Projects/smart_cctv/global_tracker.py)

A `GlobalTracker` that assigns **globally unique IDs** across cameras using appearance matching:

1. Each camera's ByteTrack assigns **local** track IDs
2. For each tracked person, extract an **HSV color histogram** from the bounding box crop (fast, lighting-invariant)
3. When a new local track appears, compare its histogram against all known global tracks using **histogram correlation**
4. **If match ≥ threshold** → reuse that global ID (same person seen from different camera)
5. **If no match** → create new global ID
6. Periodically update stored histograms to handle appearance drift

> [!NOTE]
> HSV histograms are a lightweight, model-free approach. They work well for same-room multi-camera since clothing colors are distinctive. No extra model download required.

---

#### [MODIFY] [app.py](file:///c:/Users/haris/Projects/smart_cctv/app.py)

- Instantiate one shared `GlobalTracker`
- In the camera worker loop, after ByteTrack produces local IDs, call `global_tracker.resolve(cam_id, local_id, crop)` to get the global ID
- Use global IDs for display labels and event logging

---

### Camera Config

Remove cam-01 from the `CAMERAS` dict:

```python
CAMERAS = {
    "cam-02": {"path": "videos/view-IP2.mp4", "label": "CAM_02_PARKING"},
    "cam-03": {"path": "videos/view-IP5.mp4", "label": "CAM_03_ENTRANCE"},
}
```

## Verification Plan

### Manual Verification
1. Start app, open dashboard — feed should be noticeably smoother
2. Watch a person walk across the scene on cam-02, note their ID
3. Switch to cam-03 — the same person (if visible) should have the same ID
4. Bounding boxes and labels should still be accurate
