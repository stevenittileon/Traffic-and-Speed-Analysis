# from typing import Dict, List, Set, Tuple
# import cv2
# import numpy as np
# from tqdm import tqdm
# from ultralytics import YOLO
# import supervision as sv

# from video_processing.utils import initiate_polygon_zones, COLORS, ZONE_IN_POLYGONS, ZONE_OUT_POLYGONS
# from video_processing.detections_manager import DetectionsManager


# COLORS = sv.ColorPalette.from_hex([
#     "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
#     "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
#     "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
#     "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
# ])



# class VideoProcessor:
#     def __init__(
#         self,
#         source_weights_path: str,
#         source_video_path: str,
#         target_video_path: str = None,
#         confidence_threshold: float = 0.3,
#         iou_threshold: float = 0.7,
#     ) -> None:
#         self.conf_threshold = confidence_threshold
#         self.iou_threshold = iou_threshold
#         self.source_video_path = source_video_path
#         self.target_video_path = target_video_path
#         self.frame_counter = 0
#         self.speed_update_frames = 10

#         self.model = YOLO(source_weights_path)
#         self.tracker = sv.ByteTrack()

#         self.video_info = sv.VideoInfo.from_video_path(source_video_path)
#         self.zones_in = initiate_polygon_zones(
#             ZONE_IN_POLYGONS,
#             triggering_position=sv.Position.CENTER
#         )
#         self.zones_out = initiate_polygon_zones(
#             ZONE_OUT_POLYGONS,
#             triggering_position=sv.Position.CENTER
#         )


#         self.box_annotator = sv.BoxAnnotator(color=COLORS)
#         self.trace_annotator = sv.TraceAnnotator(
#             color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
#         )
#         self.detections_manager = DetectionsManager()

#     def process_video(self):
#         frame_generator = sv.get_video_frames_generator(
#             source_path=self.source_video_path
#         )

#         if self.target_video_path:
#             with sv.VideoSink(self.target_video_path, self.video_info) as sink:
#                 for frame in tqdm(frame_generator, total=self.video_info.total_frames):
#                     annotated_frame = self.process_frame(frame)
#                     sink.write_frame(annotated_frame)
#         else:
#             for frame in tqdm(frame_generator, total=self.video_info.total_frames):
#                 annotated_frame = self.process_frame(frame)
#                 cv2.imshow("Processed Video", annotated_frame)
#                 if cv2.waitKey(1) & 0xFF == ord("q"):
#                     break
#             cv2.destroyAllWindows()

#     def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
#         annotated_frame = frame.copy()

#         frame_rate = self.video_info.fps
#         scale = 0.05  # Define the scale based on your video and real-world measurement

#         # Initialize the labels list
#         labels = []

#         for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
#             x_center = (bbox[0] + bbox[2]) / 2
#             y_center = (bbox[1] + bbox[3]) / 2
#             if self.frame_counter % self.speed_update_frames == 1:
#                 speed = self.detections_manager.calculate_speed(tracker_id, (x_center, y_center), frame_rate, scale)
#             elif self.frame_counter == 2:
#                 speed = 10*self.detections_manager.calculate_speed(tracker_id, (x_center, y_center), frame_rate, scale)
#             elif tracker_id in self.detections_manager.speeds.keys():
#                 speed = self.detections_manager.speeds[tracker_id]
#             else:
#                 speed = 0
#             labels.append(f"#{tracker_id}  Speed:{speed:.2f}km/h")

#         # Continue with annotation using the labels
#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(
#             annotated_frame, detections, labels
#         )

#         for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
#             annotated_frame = sv.draw_polygon(
#                 annotated_frame, zone_in.polygon, COLORS.colors[i]
#             )
#             annotated_frame = sv.draw_polygon(
#                 annotated_frame, zone_out.polygon, COLORS.colors[i]
#             )

#         labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
#         annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
#         annotated_frame = self.box_annotator.annotate(
#             annotated_frame, detections, labels
#         )

#         for zone_out_id, zone_out in enumerate(self.zones_out):
#             zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
#             if zone_out_id in self.detections_manager.counts:
#                 counts = self.detections_manager.counts[zone_out_id]
#                 for i, zone_in_id in enumerate(counts):
#                     count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
#                     text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
#                     annotated_frame = sv.draw_text(
#                         scene=annotated_frame,
#                         text=str(count),
#                         text_anchor=text_anchor,
#                         background_color=COLORS.colors[zone_in_id],
#                     )

#         return annotated_frame

#     def process_frame(self, frame: np.ndarray) -> np.ndarray:
#         self.frame_counter += 1
#         results = self.model(
#             frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
#         )[0]
#         detections = sv.Detections.from_ultralytics(results)
#         detections.class_id = np.zeros(len(detections))
#         detections = self.tracker.update_with_detections(detections)

#         detections_in_zones = []
#         detections_out_zones = []

#         for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
#             detections_in_zone = detections[zone_in.trigger(detections=detections)]
#             detections_in_zones.append(detections_in_zone)
#             detections_out_zone = detections[zone_out.trigger(detections=detections)]
#             detections_out_zones.append(detections_out_zone)

#         detections = self.detections_manager.update(
#             detections, detections_in_zones, detections_out_zones
#         )
#         annotated_frame = self.annotate_frame(frame, detections)
#         if self.frame_counter % self.speed_update_frames == 1:
#             self.detections_manager.update_positions(detections)
#         return annotated_frame


from typing import List
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

from video_processing.utils import initiate_polygon_zones, COLORS, ZONE_IN_POLYGONS, ZONE_OUT_POLYGONS, empty_detections
from video_processing.detections_manager import DetectionsManager


COLORS = sv.ColorPalette.from_hex([
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
])


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.frame_counter = 0

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        
        # --- SAFE POLYGON ZONES INIT ---
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS)
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS)

        self.box_annotator = sv.BoxAnnotator(color=COLORS, color_lookup=sv.ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.TOP_CENTER, text_padding=4
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    def _sanitize_class_id(self, detections: sv.Detections) -> None:
        """Ensure class_id has no None/NaN, all non-negative ints (for color lookup)."""
        if len(detections) == 0:
            return
        cid = detections.class_id
        if cid is None:
            detections.class_id = np.zeros(len(detections), dtype=np.int64)
            return
        cid = np.asarray(cid, dtype=float)
        cid = np.nan_to_num(cid, nan=0.0, posinf=0.0, neginf=0.0)
        cid = np.maximum(cid, 0).astype(np.int64)
        detections.class_id = cid

    # -----------------------------
    # VIDEO PROCESSING
    # -----------------------------
    def process_video(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    # -----------------------------
    # ANNOTATION
    # -----------------------------
    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()
        frame_rate = self.video_info.fps
        scale = 0.05  # meters per pixel (tune for your scene)

        labels = []
        if len(detections) > 0:
            for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                speed = self.detections_manager.calculate_speed(tracker_id, (x_center, y_center), frame_rate, scale)
                labels.append(f"#{tracker_id}  Speed:{speed:.2f}km/h")

        # Annotate only if detections exist
        if len(detections) > 0:
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels=labels)

        # Draw polygon zones safely
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            if zone_in and len(zone_in.polygon) > 0:
                annotated_frame = sv.draw_polygon(annotated_frame, zone_in.polygon, COLORS.colors[i % len(COLORS.colors)])
            if zone_out and len(zone_out.polygon) > 0:
                annotated_frame = sv.draw_polygon(annotated_frame, zone_out.polygon, COLORS.colors[i % len(COLORS.colors)])

        # Counts display
        for zone_out_id, zone_out in enumerate(self.zones_out):
            if not zone_out or len(zone_out.polygon) == 0:
                continue
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id % len(COLORS.colors)],
                    )

        return annotated_frame

    # -----------------------------
    # FRAME PROCESSING
    # -----------------------------
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_counter += 1
        results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)

        # --- SAFELY handle empty detections ---
        if len(detections) == 0 or detections.xyxy.size == 0:
            return frame  # skip annotation/tracking

        # Assign default class_id for tracker
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        # --- Apply zones safely ---
        detections_in_zones = []
        detections_out_zones = []

        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            # Trigger only if detections exist
            if zone_in:
                mask_in = zone_in.trigger(detections=detections)
                detections_in_zones.append(detections[mask_in] if len(mask_in) > 0 else empty_detections())
            else:
                detections_in_zones.append(empty_detections())

            if zone_out:
                mask_out = zone_out.trigger(detections=detections)
                detections_out_zones.append(detections[mask_out] if len(mask_out) > 0 else empty_detections())
            else:
                detections_out_zones.append(empty_detections())

        detections = self.detections_manager.update(detections, detections_in_zones, detections_out_zones)
        self._sanitize_class_id(detections)
        annotated_frame = self.annotate_frame(frame, detections)

        self.detections_manager.update_positions(detections)

        return annotated_frame
