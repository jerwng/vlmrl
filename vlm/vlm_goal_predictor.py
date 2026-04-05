"""
VLM Goal Predictor for Go2 Robot
Processes environment frames and predicts navigation goals (heading, velocity)
Uses Qwen/Qwen3.5-9B
"""
import json
import re

import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any

SCENE_UNDERSTANDING_PROMPT = (
    "You are guiding a robot toward the {target}.\n\n"
    "Determine:\n"
    "1. Is the {target} visible in the image?\n"
    "2. If the {target} is visible, is there any other object between the robot and the {target} that blocks the direct path to it?\n\n"
    "Important:\n"
    "- Objects on the floor can block the path.\n"
    "- Do not ignore low obstacles on the floor.\n"
    "- A blocking object is any object the robot would likely collide with if it moved directly toward the {target}.\n\n"
    "Return exactly one JSON object.\n"
    "Do not output any explanation or extra text.\n"
    "Example valid outputs:\n"
    '{{"target_visible": true, "object_blocking": true}}\n'
    '{{"target_visible": true, "object_blocking": false}}\n'
    '{{"target_visible": false, "object_blocking": false}}\n'
)


STEERING_PROMPT_TEMPLATE = (
    "You are guiding a robot toward the {target}.\n\n"
    "The direct path to the {target} is clear.\n\n"
    "Calculate a normalized heading change in [-1, 1] to steer toward the center of the {target}.\n\n"
    "Definition:\n"
    "- negative = turn left\n"
    "- positive = turn right\n"
    "- 0 = go straight\n"
    "- magnitude = how strong the turn should be\n\n"
    "Return only JSON:\n"
    '{{"heading_change": heading change value between -1 and 1}}'
)

AVOIDANCE_PROMPT_TEMPLATE = (
    "You are guiding a robot toward the {target}.\n\n"
    "The direct path to the {target} is blocked.\n\n"
    "Calculate a normalized heading change in [-1, 1] to steer toward the center of the {target}.\n\n"
    "Definition:\n"
    "- negative = turn left\n"
    "- positive = turn right\n"
    "- 0 = go straight\n"
    "- magnitude = how strong the turn should be\n\n"
    "Return only JSON:\n"
    '{{"heading_change": heading change value between -1 and 1}}'
)

# AVOIDANCE_PROMPT_TEMPLATE = (
#     "You are guiding a robot toward the {target}.\n\n"
#     "The direct path to the {target} is blocked.\n\n"
#     "Choose the safer direction to go around the blocking object and output a normalized "
#     "heading change in [-1, 1].\n\n"
#     "Rules:\n"
#     "- Choose the safer side to avoid the blocking object.\n"
#     "- Use the smallest heading change that is likely to safely go around the obstacle.\n"
#     "- Do not use a large turn unless a small turn would likely fail.\n"
#     "Definition:\n"
#     "- negative = turn left\n"
#     "- positive = turn right\n"
#     "- larger magnitude = stronger turn\n\n"
#     "Return only JSON:\n"
#     '{{ "heading_change": heading change value between -1 and 1}}'
# )

STOP_PROMPT_TEMPLATE = (
    "You are guiding a robot toward the {target}.\n\n"
    "The {target} is visible and the path is clear.\n\n"
    "Locate the {target} in the image and output its bounding box as "
    "[x1, y1, x2, y2] in a 0–1000 coordinate grid (top-left origin).\n\n"
    "Return only JSON:\n"
    '{{"bbox_2d": [x1, y1, x2, y2]}}'
)
STOP_BOTTOM_THRESHOLD = 0.80  # stop when bbox bottom >= this fraction of image height


class VLMGoalPredictor:
    """
    Base class for VLM-based goal prediction.
    Processes visual observations and outputs navigation goals.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize the VLM goal predictor.
        
        Args:
            model_path: Path to pretrained VLM model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load the VLM model.
        
        Args:
            model_path: Path to the model checkpoint
        """
        # TODO: Implement model loading logic
        # This will depend on which VLM you're using (OpenVLA, etc.)
        print(f"Loading VLM model from {model_path}...")
        print("TODO: Implement model loading")
        self.model = None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for VLM input.
        
        Args:
            frame: Raw frame from environment (H, W, C)
            
        Returns:
            Preprocessed frame ready for model input
        """
        # Convert to PIL Image for easier processing
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
        
        # TODO: Add model-specific preprocessing
        # - Resize to model input size
        # - Normalize
        # - Convert to tensor
        
        return np.array(image)
    
    def predict_goal(self, frame: np.ndarray, prompt: Optional[str] = None) -> Dict[str, float]:
        """
        Predict navigation goal from visual observation.
        
        Args:
            frame: Environment frame (H, W, C)
            prompt: Optional text prompt to guide prediction
            
        Returns:
            Dictionary with goal parameters:
                - 'vel_x': Forward/backward velocity (-1 to 1)
                - 'vel_y': Left/right velocity (-1 to 1)  
                - 'heading': Heading change in radians (-π to π)
        """
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # TODO: Implement VLM inference
        # This should:
        # 1. Pass frame (and optional prompt) to VLM
        # 2. Parse VLM output to extract navigation goals
        # 3. Convert to goal format
        
        # Placeholder: Return default forward motion
        goal = {
            'vel_x': 1.0,  # Move forward
            'vel_y': 0.0,  # No lateral movement
            'heading': 0.0  # No heading change
        }
        
        return goal
    
    def predict_heading_from_frame(self, frame: np.ndarray, prompt: Optional[str] = None) -> float:
        """
        Predict only heading from visual observation.
        Simplified version that just returns heading change.
        
        Args:
            frame: Environment frame (H, W, C)
            prompt: Optional text prompt
            
        Returns:
            Heading change in radians (-π to π)
        """
        goal = self.predict_goal(frame, prompt)
        return goal['heading']


class DummyVLMPredictor(VLMGoalPredictor):
    """
    Dummy predictor for testing without actual VLM.
    Returns predefined goals or simple heuristics.
    """
    
    def __init__(self, default_heading: float = 0.0):
        """
        Initialize dummy predictor.
        
        Args:
            default_heading: Default heading to return
        """
        super().__init__(model_path=None, device="cpu")
        self.default_heading = default_heading
        self.call_count = 0
    
    def predict_goal(self, frame: np.ndarray, prompt: Optional[str] = None, current_heading: Optional[float] = None, step: int = 0) -> Dict[str, float]:
        """
        Predict goal (dummy implementation).
        """
        self.call_count += 1
        
        # Simple heuristic: oscillate heading every 100 frames
        # This creates a simple navigation pattern for testing
        if self.call_count % 200 < 100:
            heading = 0.1  # Turn slightly right
        else:
            heading = -0.1  # Turn slightly left
        
        goal = {
            'vel_x': 1.0,
            'vel_y': 0.0,
            'heading': heading,
            'stop': False,
        }

        return goal


class HuggingFaceVLMPredictor(VLMGoalPredictor):
    """
    Qwen3.5-9B goal predictor.
    Hard-coded to use Qwen/Qwen3.5-9B (text-only model).

    Optionally loads a LoRA adapter via ``lora_adapter_path``.
    When a LoRA adapter is loaded, the predictor uses the multi-object
    detection prompt (DETECTION_PROMPT) and exposes all detected objects
    via ``self.last_detections`` after each ``predict_goal()`` call.

"""

    def __init__(self, model_path: str = None, device: str = "cuda",
                 model_type: str = "auto", trust_remote_code: bool = True,
                 lora_adapter_path: Optional[str] = None):
        """
        Initialize Qwen3.5-9B predictor.

        Args:
            model_path: Ignored (always uses Qwen/Qwen3.5-9B)
            device: Device to run inference on ('cuda' or 'cpu')
            model_type: Ignored (kept for compatibility)
            trust_remote_code: Whether to trust remote code
            lora_adapter_path: Optional path to a LoRA adapter directory.
                If provided, the adapter is merged into the base model at
                load time via PeftModel.from_pretrained() + merge_and_unload().
        """
        self.model_type = model_type
        self.trust_remote_code = trust_remote_code
        self.lora_adapter_path = lora_adapter_path
        self.last_detections: List[dict] = []
        self.previous_goal_heading = None  # Track previous heading for u=0 case
        super().__init__(model_path or "Qwen/Qwen3.5-9B", device)

    def load_model(self, model_path: str):
        """
        Load Qwen3.5-9B model and tokenizer.

        Args:
            model_path: Ignored, using hard-coded Qwen/Qwen3.5-9B
        """
        # Hard-coded model path
        model_path = "Qwen/Qwen3.5-9B"
        self.model_path = model_path

        print(f"Loading Qwen3.5-9B model: {model_path}")

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            # Load processor
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code

            )

            # Load model
            print(f"Loading model onto {self.device}...")
            is_cuda = self.device == "cuda" or self.device.startswith("cuda:")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.bfloat16 if is_cuda else torch.float32,
                device_map={"": self.device} if is_cuda else "cpu"
            )

            self.model.eval()
            print(f"Model loaded successfully!")

            # Optionally merge LoRA adapter
            if self.lora_adapter_path is not None:
                try:
                    from peft import PeftModel
                    print(f"Loading LoRA adapter from: {self.lora_adapter_path}")
                    self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
                    self.model = self.model.merge_and_unload()
                    self.model.eval()
                    print("LoRA adapter merged successfully.")
                except ImportError:
                    raise ImportError(
                        "peft is required for LoRA inference. Install with: pip install peft"
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load LoRA adapter from {self.lora_adapter_path}: {e}"
                    )

        except ImportError as e:
            print(f"ERROR: Required library not found. Install with: pip install transformers qwen-vl-utils")
            print(f"Error details: {e}")
            raise
        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Preprocess frame for HF model input.
        
        Args:
            frame: Raw frame from environment (H, W, C)
            
        Returns:
            PIL Image ready for processor
        """
        if isinstance(frame, np.ndarray):
            print(f"Input frame shape: {frame.shape}, dtype: {frame.dtype}")
            image = Image.fromarray(frame)
        else:
            image = frame
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"Preprocessed image size: {image.size}, mode: {image.mode}")
        return image
    
    def query_model(self, image: Image.Image, prompt: str) -> str:
        """
        Query Qwen3-VL with an image and text prompt.
        
        Args:
            image: PIL Image
            prompt: Text prompt
            
        Returns:
            Model's text response
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        from qwen_vl_utils import process_vision_info
        
        # Prepare messages in Qwen3-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, 
            enable_thinking=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        if self.device == "cuda" or self.device.startswith("cuda:"):
            inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                repetition_penalty=1.0,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        think_match = re.search(r'(.*?)</think>\n?', response, flags=re.DOTALL)
        if think_match:
            print(f"[VLM Thinking]: {think_match.group(1).strip()}")
            response = response[think_match.end():].strip()
        
        return response
    
    def parse_response_to_goal(self, response: str, current_heading: float = 0.0) -> Dict[str, float]:
        """
        Parse VLM response into navigation goals.

        Expects JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]
        Coordinates are in [0, 1000] (normalized to model image grid).

        Args:
            response: Text response from VLM
            current_heading: Current robot heading in radians

        Returns:
            Goal dictionary with vel_x, vel_y, heading (absolute), relative_heading, stop
        """
        import re

        goal = {
            'vel_x': 1.0,
            'vel_y': 0.0,
            'heading': current_heading,
            'relative_heading': 0.0,
            'stop': False,
        }

        # Parse JSON bbox_2d format: {"bbox_2d": [x1, y1, x2, y2], ...}
        bbox_match = re.search(r'"bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response)

        if bbox_match is None:
            print("Red disk not detected in grounding output — stopping")

            goal['vel_x'] = 0.0
            goal['vel_y'] = 0.0
            goal['stop'] = True
            return goal

        x1, y1, x2, y2 = (int(bbox_match.group(i)) for i in range(1, 5))
        print(f"Detected bbox: ({x1},{y1}),({x2},{y2})")

        # Compute horizontal center in [0, 1000] and map to u in [-1, 1]
        cx = (x1 + x2) / 2.0
        u = (cx - 500.0) / 500.0
        u = np.clip(u, -1.0, 1.0)

        # Compute bbox area fraction to decide stop
        bottom_fraction = y2 / 1000.0
        print(f"Bbox bottom fraction: {bottom_fraction:.4f} (threshold={STOP_BOTTOM_THRESHOLD})")

        if bottom_fraction >= STOP_BOTTOM_THRESHOLD:
            print("Red disk is NEAR (large bbox) — signaling STOP")
            goal['vel_x'] = 0.0
            goal['vel_y'] = 0.0
            goal['relative_heading'] = 0.0
            goal['heading'] = current_heading
            goal['stop'] = True
            return goal

        # Disk is visible but far: steer toward it
        relative_heading_radians = -u * 45.0 * np.pi / 180.0
        absolute_heading = current_heading + relative_heading_radians

        goal['relative_heading'] = relative_heading_radians
        goal['vel_x'] = 1.0
        goal['stop'] = False

        if u == 0.0 and self.previous_goal_heading is not None:
            goal['heading'] = self.previous_goal_heading
        else:
            goal['heading'] = absolute_heading

        return goal

    def parse_detections_response(self, response: str) -> List[dict]:
        """
        Parse a JSON array of detections from the finetuned model's response.

        Expects format: [{"bbox": [x1, y1, x2, y2], "label": "..."}, ...]

        Returns:
            List of validated detection dicts. Returns [] on any parse failure.
        """
        detections = None

        # Try direct parse first
        try:
            detections = json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to extract a JSON array substring via regex
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                try:
                    detections = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if detections is None or not isinstance(detections, list):
            print(
                f"WARNING: Failed to parse detection response: {response[:200]}"
            )
            return []

        # Validate each item
        validated = []
        for item in detections:
            if not isinstance(item, dict):
                print(f"WARNING: Detection item is not a dict: {item}")
                continue
            bbox = item.get("bbox")
            label = item.get("label")
            if not isinstance(bbox, list) or len(bbox) != 4:
                print(f"WARNING: Malformed bbox in detection: {item}")
                continue
            if not isinstance(label, str):
                print(f"WARNING: Missing or non-string label in detection: {item}")
                continue
            validated.append({"bbox": [int(v) for v in bbox], "label": label})

        return validated

    def predict_goal(self, frame: np.ndarray, prompt: Optional[str] = None, current_heading: float = 0.0) -> Dict[str, float]:
        """
        Predict navigation goal from visual observation using Qwen3-VL.

        When ``lora_adapter_path`` is set, uses the multi-object DETECTION_PROMPT,
        parses ALL detected objects, stores them in ``self.last_detections``, and
        steers toward the first detected object using the standard u-based formula.

        When no LoRA adapter is loaded, uses the existing red-disk detection logic
        (unchanged from the original implementation).

        Args:
            frame: Environment frame (H, W, C)
            prompt: Text prompt to guide prediction
            current_heading: Current robot heading in radians (yaw angle)

        Returns:
            Dictionary with goal parameters (including absolute and relative heading)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess frame
        image = self.preprocess_frame(frame)

        # LoRA path: multi-object detection
        if self.lora_adapter_path is not None:
            if prompt is None:
                from vlmrl.vlm.finetune.dataset import DETECTION_PROMPT
                prompt = DETECTION_PROMPT

            response = self.query_model(image, prompt)
            print("*" * 50)
            print(f"VLM Response (LoRA): {response}")

            # Parse all detections; store on self for downstream consumers
            self.last_detections = self.parse_detections_response(response)
            print(f"Detections parsed: {len(self.last_detections)} object(s)")

            if not self.last_detections:
                print("No objects detected — stopping")
                goal = {
                    "vel_x": 0.0,
                    "vel_y": 0.0,
                    "heading": current_heading,
                    "relative_heading": 0.0,
                    "stop": True,
                }
            else:
                # Steer toward the first detected object's bbox center
                first_bbox = self.last_detections[0]["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = first_bbox
                cx = (x1 + x2) / 2.0
                u = np.clip((cx - 500.0) / 500.0, -1.0, 1.0)
                relative_heading_radians = -u * 45.0 * np.pi / 180.0
                absolute_heading = current_heading + relative_heading_radians
                goal = {
                    "vel_x": 1.0,
                    "vel_y": 0.0,
                    "heading": absolute_heading,
                    "relative_heading": relative_heading_radians,
                    "stop": False,
                }
                print(f"Steering toward '{self.last_detections[0]['label']}' bbox={first_bbox} u={u:.3f}")

            print(f"Parsed Goal: {goal}")
            print("*" * 50)

            if "heading" in goal:
                self.previous_goal_heading = goal["heading"]

            return goal

        # Original path: red-disk detection (no LoRA)
        if prompt is None:
            prompt = (
                "Locate the red circular disk on the ground in this image. "
                "Output its bounding box. "
                "If no red disk is visible, output \"not visible\"."
            )

        # Query model
        response = self.query_model(image, prompt)
        print("*" * 50)
        print(f"Prompt: {prompt}")
        print(f"VLM Response: {response}")

        # Parse response to goal
        goal = self.parse_response_to_goal(response, current_heading=current_heading)

        print(f"Parsed Goal: {goal}")
        print("*" * 50)

        # Store current goal heading for next prediction
        if "heading" in goal:
            self.previous_goal_heading = goal["heading"]

        return goal


class TwoPassVLMPredictor(HuggingFaceVLMPredictor):
    """
    Two-pass VLM predictor for goal-directed navigation.

    Pass 1: Localize the target object via bounding box detection.
    Pass 2: Predict a heading with obstruction awareness.

    Inherits model loading, frame preprocessing, and query_model from
    HuggingFaceVLMPredictor. Does not support LoRA adapters.
    """

    # Assumed deceleration (m/s^2) used to estimate stopping distance
    _DECEL_MPS2 = 0.25

    def __init__(self, target_object: str = "red disk marker",
                 model_path: str = None, device: str = "cuda",
                 model_type: str = "auto", trust_remote_code: bool = True):
        # Explicitly pass lora_adapter_path=None to disable LoRA in parent
        super().__init__(
            model_path=model_path,
            device=device,
            model_type=model_type,
            trust_remote_code=trust_remote_code,
            lora_adapter_path=None,
        )
        self.target_object = target_object
        self.vlm_time_total = 0.0   # cumulative VLM inference wall time (seconds)
        self.vlm_call_count = 0     # number of predict_goal calls

    def _parse_scene_response(self, response: str) -> Optional[dict]:
        """Parse a Pass 1 scene understanding response.

        Returns a dict with keys:
            target_visible (bool),
            object_blocking (bool)
        or None on failure.
        """
        try:
            result = json.loads(response.strip())
            if isinstance(result, dict):
                return {
                    "target_visible": bool(result.get("target_visible", False)),
                    "object_blocking": bool(result.get("object_blocking", False)),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback
        vis_match = re.search(r'"target_visible"\s*:\s*(true|false)', response, re.IGNORECASE)
        blk_match = re.search(r'"object_blocking"\s*:\s*(true|false)', response, re.IGNORECASE)
        if vis_match and blk_match:
            return {
                "target_visible": (vis_match.group(1).lower() == "true"),
                "object_blocking": (blk_match.group(1).lower() == "true")
            }

        print(f"WARNING: _parse_scene_response failed to parse: {response[:200]}")
        return None

    def _parse_heading_response(self, response: str) -> Optional[float]:
        """Parse a Pass 2 response into (heading_change_float, avoid_direction_str) or (None, "left")."""
        try:
            result = json.loads(response.strip())
            if isinstance(result, dict) and "heading_change" in result:
                heading = float(np.clip(float(result["heading_change"]), -1.0, 1.0))
                return heading
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback
        h_match = re.search(r'"heading_change"\s*:\s*(-?[0-9]*\.?[0-9]+)', response)
        if h_match:
            heading = float(np.clip(float(h_match.group(1)), -1.0, 1.0))
            return heading

        print(f"WARNING: _parse_heading_response failed to parse: {response[:200]}")
        return  None

    def _parse_stop_response(self, response: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse a Pass 3 stop response into a bbox tuple (x1, y1, x2, y2) or None.

        Returns the bounding box if successfully parsed, None on failure.
        """
        try:
            result = json.loads(response.strip())
            if isinstance(result, dict):
                bbox = result.get("bbox_2d")
                if isinstance(bbox, list) and len(bbox) == 4:
                    return tuple(int(v) for v in bbox)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Regex fallback
        match = re.search(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response)
        if match:
            return tuple(int(match.group(i)) for i in range(1, 5))

        print(f"WARNING: _parse_stop_response failed to parse bbox: {response[:200]}")
        return None

    def predict_goal(self, frame: np.ndarray, prompt: Optional[str] = None,
                     current_heading: float = 0.0) -> Dict[str, Any]:
        """
        Run the three-pass VLM pipeline and return a goal dict with keys:
        vel_x, vel_y, heading, relative_heading, stop, obstructed.

        Pass 1: Scene understanding (target visible? path blocked?)
        Pass 2: Heading prediction (steer or avoid)
        Pass 3: Stop check via bbox area fraction — only runs when path is NOT blocked.

        Args:
            frame: Egocentric camera frame (H, W, C).
            prompt: Unused (kept for interface compatibility).
            current_heading: Current robot yaw in radians.
        """
        import time

        _N_SAMPLES = 3
        image = self.preprocess_frame(frame)

        # ---- Pass 1: Scene Understanding (3 samples, majority vote) ----
        t0 = time.time()
        pass1_prompt = SCENE_UNDERSTANDING_PROMPT.format(target=self.target_object)
        pass1_scenes = []
        for i in range(_N_SAMPLES):
            resp = self.query_model(image, pass1_prompt)
            parsed = self._parse_scene_response(resp)
            print(f"[Pass 1 Sample {i+1}] Response: {resp}")
            print(f"[Pass 1 Sample {i+1}] Parsed: {parsed}")
            if parsed is not None:
                pass1_scenes.append(parsed)
        t1 = time.time()
        self.vlm_time_total += t1 - t0
        self.vlm_call_count += 1
        print("*" * 50)
        print(f"[Pass 1] Wall time: {t1 - t0:.2f}s  ({len(pass1_scenes)}/{_N_SAMPLES} parsed)")

        if not pass1_scenes:
            print("[Pass 1] All samples failed to parse — stopping")
            goal = {
                "vel_x": 0.0, "vel_y": 0.0,
                "heading": current_heading, "relative_heading": 0.0,
                "stop": True, "obstructed": False,
            }
            print(f"[Final Goal]: {goal}")
            print("*" * 50)
            self.previous_goal_heading = current_heading
            return goal

        # Majority vote on boolean fields
        target_visible = sum(s["target_visible"] for s in pass1_scenes) > len(pass1_scenes) / 2
        object_blocking = sum(s["object_blocking"] for s in pass1_scenes) > len(pass1_scenes) / 2
        scene = {"target_visible": target_visible, "object_blocking": object_blocking}
        print(f"[Pass 1 Aggregated]: {scene}")

        if not scene["target_visible"]:
            print("[Pass 1] Target not visible (majority) — stopping")
            goal = {
                "vel_x": 0.0, "vel_y": 0.0,
                "heading": current_heading, "relative_heading": 0.0,
                "stop": True, "obstructed": False,
            }
            print(f"[Final Goal]: {goal}")
            print("*" * 50)
            self.previous_goal_heading = current_heading
            return goal

        # ---- Pass 2: Heading Prediction (3 samples, median) ----
        if not scene["object_blocking"]:
            pass2_prompt = STEERING_PROMPT_TEMPLATE.format(target=self.target_object)
        else:
            pass2_prompt = AVOIDANCE_PROMPT_TEMPLATE.format(target=self.target_object)
        t2 = time.time()
        pass2_headings = []
        for i in range(_N_SAMPLES):
            resp = self.query_model(image, pass2_prompt)
            parsed = self._parse_heading_response(resp)
            print(f"[Pass 2 Sample {i+1}] Response: {resp}")
            print(f"[Pass 2 Sample {i+1}] Parsed: {parsed}")
            if parsed is not None:
                pass2_headings.append(parsed)
        t3 = time.time()
        self.vlm_time_total += t3 - t2
        print(f"[Pass 2] Wall time: {t3 - t2:.2f}s  ({len(pass2_headings)}/{_N_SAMPLES} parsed)")

        if not pass2_headings:
            print("[Pass 2] All samples failed to parse — defaulting heading_raw=0.0")
            heading_raw = 0.0
        else:
            heading_raw = float(np.median(pass2_headings))
        print(f"[Pass 2 Aggregated heading_raw]: {heading_raw}")

        # Dead-zone
        if abs(heading_raw) < 0.1:
            heading_raw = 0.0

        relative_heading_radians = -heading_raw * 45.0 * np.pi / 180.0
        absolute_heading = current_heading + relative_heading_radians

        if heading_raw == 0.0 and self.previous_goal_heading is not None:
            absolute_heading = self.previous_goal_heading

        if scene["object_blocking"]:
            vel_x = 1.0 - 0.65 * abs(heading_raw)
            print(f"[Blocking] Scaling vel_x: 1.0 - 0.65 * {abs(heading_raw):.3f} = {vel_x:.3f}")
        else:
            vel_x = 1.0

        # ---- Pass 3: Stop Check (bbox area fraction) — unblocked path only ----
        should_stop = False
        if not scene["object_blocking"]:
            t4 = time.time()
            pass3_prompt = STOP_PROMPT_TEMPLATE.format(target=self.target_object)
            pass3_stops = []
            for i in range(_N_SAMPLES):
                resp = self.query_model(image, pass3_prompt)
                print(f"[Pass 3 Sample {i+1}] Response: {resp}")
                stop_bbox = self._parse_stop_response(resp)
                print(f"[Pass 3 Sample {i+1}] Bbox: {stop_bbox}")
                if stop_bbox is not None:
                    _x1, _y1, _x2, y2 = stop_bbox
                    bottom_fraction = y2 / 1000.0
                    print(f"[Pass 3 Sample {i+1}] Bbox bottom fraction: {bottom_fraction:.4f} (threshold={STOP_BOTTOM_THRESHOLD})")
                    pass3_stops.append(bottom_fraction >= STOP_BOTTOM_THRESHOLD)
                else:
                    print(f"[Pass 3 Sample {i+1}] No bbox parsed — treating as not stop")
                    pass3_stops.append(False)
            t5 = time.time()
            self.vlm_time_total += t5 - t4
            print(f"[Pass 3] Wall time: {t5 - t4:.2f}s  ({len(pass3_stops)}/{_N_SAMPLES} parsed)")
            print(f"[Pass 3 Stop votes]: {pass3_stops}")
            should_stop = sum(pass3_stops) > len(pass3_stops) / 2
        else:
            print("[Pass 3] Skipped (path blocked — avoidance active)")

        print(f"[Pass 3 Stop]: {should_stop}")

        if should_stop:
            print("[Pass 3] Target near — zeroing velocity, holding heading")
            goal = {
                "vel_x": 0.0, "vel_y": 0.0,
                "heading": current_heading,
                "relative_heading": 0.0,
                "stop": True,
            }
            print(f"[Final Goal]: {goal}")
            print("*" * 50)
            self.previous_goal_heading = current_heading
            return goal

        goal = {
            "vel_x": vel_x, "vel_y": 0.0,
            "heading": absolute_heading,
            "relative_heading": relative_heading_radians,
            "stop": False,
        }
        print(f"[Final Goal]: {goal}")
        print("*" * 50)
        self.previous_goal_heading = absolute_heading
        return goal


class OpenVLAGoalPredictor(VLMGoalPredictor):
    """
    OpenVLA-based goal predictor (kept for compatibility).
    Note: Consider using HuggingFaceVLMPredictor instead.
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        print("WARNING: OpenVLAGoalPredictor is deprecated. Consider using HuggingFaceVLMPredictor.")
        super().__init__(model_path, device)
    
    def load_model(self, model_path: str):
        """
        Load OpenVLA model.
        """
        print(f"Loading OpenVLA model from {model_path}...")
        # TODO: Implement OpenVLA model loading if needed
        # from openvla import load_vla
        # self.model = load_vla(model_path, device=self.device)
        raise NotImplementedError("OpenVLA integration not implemented. Use HuggingFaceVLMPredictor instead.")
    
    def predict_goal(self, frame: np.ndarray, prompt: Optional[str] = None) -> Dict[str, float]:
        """
        Use OpenVLA to predict navigation goal.
        """
        raise NotImplementedError("OpenVLA integration not implemented. Use HuggingFaceVLMPredictor instead.")


def create_vlm_predictor(predictor_type: str = "dummy", **kwargs) -> VLMGoalPredictor:
    """
    Factory function to create VLM predictors.

    Args:
        predictor_type: Type of predictor ('dummy', 'huggingface', 'hf', 'twopass', 'openvla')
        **kwargs: Additional arguments for predictor initialization
            - model_path: Ignored for 'huggingface'/'twopass' (always uses Qwen3-VL-8B-Instruct)
            - device: 'cuda' or 'cpu' (default: 'cuda')
            - trust_remote_code: Trust remote code (default: True)
            - target_object: str (default: 'red disk marker') — used only with 'twopass'

    Returns:
        VLMGoalPredictor instance

    Examples:
        # Dummy predictor for testing
        predictor = create_vlm_predictor('dummy')

        # Qwen3-VL-8B-Instruct (single-pass)
        predictor = create_vlm_predictor('huggingface', device='cuda')

        # Three-pass predictor with configurable target
        predictor = create_vlm_predictor('twopass', device='cuda', target_object='red disk marker')
    """
    if predictor_type == "dummy":
        return DummyVLMPredictor(**kwargs)
    elif predictor_type in ["huggingface", "hf"]:
        return HuggingFaceVLMPredictor(**kwargs)
    elif predictor_type == "twopass":
        target_object = kwargs.pop("target_object", "red disk marker")
        return TwoPassVLMPredictor(target_object=target_object, **kwargs)
    elif predictor_type == "openvla":
        return OpenVLAGoalPredictor(**kwargs)
    else:
        raise ValueError(
            f"Unknown predictor type: {predictor_type}. "
            "Choose from: dummy, huggingface, hf, twopass, openvla"
        )
