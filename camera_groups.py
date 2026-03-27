import json
import os

_GROUPS_FILE = os.path.join(os.path.dirname(__file__), "camera_groups.json")
_GROUPS = []

def _load():
    global _GROUPS
    try:
        with open(_GROUPS_FILE, "r") as f:
            data = json.load(f)
            _GROUPS = data.get("groups", [])
    except Exception as e:
        print(f"⚠ camera_groups: could not load {_GROUPS_FILE}: {e}")
        _GROUPS = []

_load()

def get_all_groups():
    return _GROUPS

def get_group(group_id):
    for g in _GROUPS:
        if g["group_id"] == group_id:
            return g
    return None

def get_cameras_for_group(group_id):
    g = get_group(group_id)
    return g["cameras"] if g else []

def get_group_for_camera(cam_id):
    cam_id_str = str(cam_id)
    for g in _GROUPS:
        if cam_id_str in g["cameras"]:
            return g
    return None

def resolve_scope(scope):
    """Given a floor, zone, or group_id string, return the list of cam_ids it resolves to."""
    if not scope:
        return []
    # Try direct group_id match first
    g = get_group(scope)
    if g:
        return g["cameras"]
    # Try floor match
    cams = set()
    for g in _GROUPS:
        if g.get("floor") == scope or g.get("zone") == scope:
            cams.update(g["cameras"])
    return list(cams)
