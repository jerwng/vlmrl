import re
import os
import sys
import numpy as np
import cv2
from typing import Optional, Tuple, Dict

from vlm_goal_predictor import HuggingFaceVLMPredictor

# Add vlmrl/ parent to sys.path for utils imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.draw_bbox import draw_bbox_on_frame


class BBoxVLMPredictor:
    """
    Wrapper around HuggingFaceVLMPredictor that extracts bounding box
    coordinates from VLM responses and annotates frames with bbox overlays.
    """

    def __init__(self, wrapped_predictor: HuggingFaceVLMPredictor):
        self.wrapped = wrapped_predictor
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_u_value: Optional[float] = None
        self.last_raw_response: Optional[str] = None

    def _extract_bbox(self, response: str) -> Optional[Tuple[int, int, int, int]]:
        match = re.search(r'"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response)
        if match is None:
            return None
        return tuple(int(match.group(i)) for i in range(1, 5))

    def _compute_u_from_bbox(self, bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        u = (cx - 500.0) / 500.0
        return float(np.clip(u, -1.0, 1.0))

    def predict_and_annotate(
        self,
        frame: np.ndarray,
        prompt: Optional[str] = None,
        current_heading: float = 0.0,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Run VLM prediction and return (goal_dict, annotated_frame).
        """
        image = self.wrapped.preprocess_frame(frame)

        if prompt is None:
            prompt = (
                "Locate the red circular disk on the ground in this image. "
                "Output its bounding box. "
                "If no red disk is visible, output \"not visible\"."
            )

        response = self.wrapped.query_model(image, prompt)
        goal = self.wrapped.parse_response_to_goal(response, current_heading=current_heading)

        # Update previous_goal_heading to match predict_goal() behavior
        if 'heading' in goal:
            self.wrapped.previous_goal_heading = goal['heading']

        # Extract bbox and u from raw response
        self.last_raw_response = response
        self.last_bbox = self._extract_bbox(response)
        self.last_u_value = self._compute_u_from_bbox(self.last_bbox) if self.last_bbox is not None else None

        annotated = draw_bbox_on_frame(frame, self.last_bbox, u_value=self.last_u_value)
        return goal, annotated

    def predict_and_save(
        self,
        frame: np.ndarray,
        output_path: str,
        prompt: Optional[str] = None,
        current_heading: float = 0.0,
    ) -> Dict[str, float]:
        """
        Run VLM prediction, save annotated frame to disk, return goal dict.
        """
        goal, annotated = self.predict_and_annotate(frame, prompt, current_heading)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr)
        return goal
