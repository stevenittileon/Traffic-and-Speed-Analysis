from typing import Dict, List, Set, Tuple
import numpy as np
import supervision as sv


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}
        self.previous_positions: Dict[int, Tuple[float, float]] = {}  # Tracker ID to (x, y)
        self.speeds: Dict[int, float] = {}  # Tracker ID to speed

    def update_positions(self, detections: sv.Detections):
        tid = getattr(detections, "tracker_id", None)
        xy = getattr(detections, "xyxy", None)
        if tid is None or xy is None or len(tid) == 0 or len(xy) == 0:
            return
        for tracker_id, bbox in zip(np.atleast_1d(tid), xy):
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            self.previous_positions[int(tracker_id)] = (x_center, y_center)

    def calculate_speed(self, tracker_id, new_position, frame_rate, scale):
        tid = int(tracker_id)
        if tid not in self.previous_positions:
            return 0.0
        old = self.previous_positions[tid]
        dx = new_position[0] - old[0]
        dy = new_position[1] - old[1]
        distance_pixels = np.sqrt(dx * dx + dy * dy)
        if distance_pixels < 1e-6:
            return float(self.speeds.get(tid, 0.0))  # reuse last speed when nearly static
        # 1-frame delta: distance (m) * fps = m/s; * 3.6 = km/h
        m_per_pixel = scale
        speed_ms = distance_pixels * m_per_pixel * frame_rate
        speed_kmh = speed_ms * 3.6
        self.speeds[tid] = float(speed_kmh)
        return self.speeds[tid] 

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        # No zones: show all detections (class_id 0), no filtering
        if not detections_in_zones and not detections_out_zones:
            detections_all.class_id = np.zeros(len(detections_all), dtype=int)
            return detections_all

        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            tid = getattr(detections_in_zone, "tracker_id", None)
            if tid is not None:
                for tracker_id in np.atleast_1d(tid):
                    self.tracker_id_to_zone_id.setdefault(int(tracker_id), zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            tid = getattr(detections_out_zone, "tracker_id", None)
            if tid is not None:
                for tracker_id in np.atleast_1d(tid):
                    tracker_id = int(tracker_id)
                    if tracker_id in self.tracker_id_to_zone_id:
                        zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                        self.counts.setdefault(zone_out_id, {})
                        self.counts[zone_out_id].setdefault(zone_in_id, set())
                        self.counts[zone_out_id][zone_in_id].add(tracker_id)

        detections_all.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(int(x), -1)
        )(detections_all.tracker_id)
        return detections_all[detections_all.class_id != -1]

