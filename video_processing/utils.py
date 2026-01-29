# import numpy as np
# import supervision as sv
# from typing import List, Tuple

# COLORS = sv.ColorPalette.from_hex([
#     "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
#     "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
#     "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
#     "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
# ])

# # ZONE_IN_POLYGONS = [
# #     np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
# #     np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
# #     np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
# #     np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
# # ]

# # ZONE_OUT_POLYGONS = [
# #     np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
# #     np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
# #     np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
# #     np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
# # ]

# ZONE_IN_POLYGONS = []
# ZONE_OUT_POLYGONS = []

# def initiate_polygon_zones(
#     polygons: List[np.ndarray],
#     triggering_position: sv.Position = sv.Position.CENTER,
# ) -> List[sv.PolygonZone]:
#     """
#     Initialize polygon zones safely for the current supervision API.

#     Args:
#         polygons: List of polygons, each as an ndarray of shape (N, 2) representing points.
#         triggering_position: Position used for triggering (default CENTER).

#     Returns:
#         List of PolygonZone objects (empty list if polygons is empty).
#     """
#     if not polygons:
#         return []  # Safely return empty if no polygons provided

#     polygon_zones = []
#     for polygon in polygons:
#         # Skip empty polygons
#         if polygon is None or len(polygon) == 0:
#             continue

#         # Create PolygonZone using current supervision API
#         zone = sv.PolygonZone(
#             polygon=polygon,
#             triggering_position=triggering_position
#         )
#         polygon_zones.append(zone)

#     return polygon_zones


import numpy as np
import supervision as sv
from typing import List

# -----------------------------
# COLOR PALETTE
# -----------------------------
COLORS = sv.ColorPalette.from_hex([
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
])

# -----------------------------
# POLYGON ZONES
# -----------------------------
# Temporarily empty; prevents crashes until you define your actual zones
ZONE_IN_POLYGONS = []
ZONE_OUT_POLYGONS = []

# Example polygons (optional, uncomment and adjust coordinates)
# ZONE_IN_POLYGONS = [
#     np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
#     np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
# ]
# ZONE_OUT_POLYGONS = [
#     np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
# ]

# -----------------------------
# INITIATE POLYGON ZONES
# -----------------------------
def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    """
    Initialize polygon zones safely for the current supervision API.

    Args:
        polygons: List of polygons, each as an ndarray of shape (N, 2) representing points.
        triggering_position: Position used for triggering (default CENTER).

    Returns:
        List of PolygonZone objects (empty list if polygons is empty).
    """
    if not polygons:
        return []

    polygon_zones = []
    for polygon in polygons:
        if polygon is None or len(polygon) == 0:
            continue

        try:
            zone = sv.PolygonZone(
                polygon=polygon,
                triggering_position=triggering_position
            )
            polygon_zones.append(zone)
        except Exception as e:
            print(f"[WARNING] Failed to create PolygonZone: {e}")
            continue

    return polygon_zones


def empty_detections() -> sv.Detections:
    """Return a valid empty sv.Detections instance."""
    try:
        return sv.Detections.empty()
    except Exception:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )
