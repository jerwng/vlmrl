"""
VLM Goal Predictor for Go2 Robot
Processes environment frames and predicts navigation goals (heading, velocity)
Uses Qwen/Qwen3-VL-8B-Instruct
"""
import numpy as np
from typing import Dict, Optional, Tuple, Any
from PIL import Image
import torch


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
    Qwen3-VL-8B-Instruct goal predictor.
    Hard-coded to use Qwen/Qwen3-VL-8B-Instruct.
    """
    
    def __init__(self, model_path: str = None, device: str = "cuda", 
                 model_type: str = "auto", trust_remote_code: bool = True):
        """
        Initialize Qwen3-VL predictor.
        
        Args:
            model_path: Ignored (always uses Qwen/Qwen3-VL-8B-Instruct)
            device: Device to run inference on ('cuda' or 'cpu')
            model_type: Ignored (kept for compatibility)
            trust_remote_code: Whether to trust remote code
        """
        self.model_type = model_type
        self.trust_remote_code = trust_remote_code
        self.previous_goal_heading = None  # Track previous heading for u=0 case
        super().__init__(model_path or "Qwen/Qwen3-VL-8B-Instruct", device)
    
    def load_model(self, model_path: str):
        """
        Load Qwen3-VL-8B-Instruct model and processor.
        
        Args:
            model_path: Ignored, using hard-coded Qwen/Qwen3-VL-8B-Instruct
        """
        # Hard-coded model path
        model_path = "Qwen/Qwen3-VL-8B-Instruct"
        self.model_path = model_path
        
        print(f"Loading Qwen3-VL model: {model_path}")
        
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            
            # Load processor
            print("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code
            )
            
            # Load model with Qwen3VL's native class
            print(f"Loading model onto {self.device}...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else "cpu"
            )
            
            self.model.eval()
            print(f"Model loaded successfully!")
            
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
            messages, tokenize=False, add_generation_prompt=True
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
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,            # no truncation
                repetition_penalty=1.0
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    def parse_response_to_goal(self, response: str, current_heading: float = 0.0) -> Dict[str, float]:
        """
        Parse VLM text response into navigation goals.
        
        Args:
            response: Text response from VLM
            current_heading: Current robot heading in radians (for calculating absolute heading)            
        Returns:
            Goal dictionary with vel_x, vel_y, heading (absolute), relative_heading
        """
        import re
        import json
        
        goal = {
            'vel_x': 1.0,   # Default forward
            'vel_y': 0.0,
            'heading': current_heading,  # Absolute heading
            'stop': False,
        }

        # Try JSON parsing (preferred format)
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
            print(f"json match 1 {json_match}")
            if not json_match:
                json_match = re.search(r'{.*?}', response, re.DOTALL)
            print(f"json match 2 {json_match}")

            if json_match:
                json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
                data = json.loads(json_str)

                # New format: {"u": <number or null>, "stop": <bool>}
                if 'u' in data:
                    stop = bool(data.get("stop", False))

                    # Handle stop=true: halt the robot
                    if stop:
                        print("VLM signaled STOP — setting vel_x=0.0")
                        goal['vel_x'] = 0.0
                        goal['vel_y'] = 0.0
                        goal['relative_heading'] = 0.0
                        goal['heading'] = current_heading
                        goal['stop'] = True
                        return goal

                    u_raw = data['u']

                    # Handle null (Python None) or NaN: target not visible
                    if u_raw is None:
                        print("Target not visible (u=null), maintaining current heading")
                        goal['relative_heading'] = 0.0
                        goal['heading'] = current_heading
                        goal['vel_x'] = 1.0
                        goal['stop'] = False
                        return goal

                    u = float(u_raw)

                    # Handle NaN case (target not visible)
                    if np.isnan(u):
                        print("Target not visible (u=nan), maintaining current heading")
                        goal['relative_heading'] = 0.0
                        goal['heading'] = current_heading
                        goal['vel_x'] = 1.0  # Keep moving forward
                        goal['stop'] = False
                        return goal

                    # Clamp u to [-1, 1]
                    u = np.clip(u, -1.0, 1.0)

                    # if np.abs(u) < 0.1:
                    #     u = 0.0

                    # Calculate relative heading: u * 45 degrees
                    relative_heading_degrees = -u * 45.0
                    relative_heading_radians = relative_heading_degrees * np.pi / 180.0

                    # Calculate absolute heading: current_heading + relative_heading
                    absolute_heading = current_heading + relative_heading_radians

                    goal['relative_heading'] = relative_heading_radians
                    goal['vel_x'] = 1.0  # Keep moving forward
                    goal['stop'] = False

                    # Special case: u == 0, use previous heading if available
                    if u == 0.0 and self.previous_goal_heading is not None:
                        goal['heading'] = self.previous_goal_heading
                    else:
                        goal['heading'] = absolute_heading
                    return goal

        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
            # Fall back to default
            print(f"Warning: JSON parsing failed ({e}), using default forward motion")

        # Fallback: maintain current heading, move forward
        goal['relative_heading'] = 0.0
        goal['heading'] = current_heading
        goal['vel_x'] = 1.0
        goal['stop'] = False

        return goal
    
    def predict_goal(self, frame: np.ndarray, prompt: Optional[str] = None, current_heading: float = 0.0) -> Dict[str, float]:
        """
        Predict navigation goal from visual observation using Qwen3-VL.
        
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
        
        # Default prompt if none provided
        if prompt is None:
            prompt = (
                "You are the navigation controller for a quadruped robot.\n"
                "You see ONE egocentric RGB image from the robot's head camera.\n"
                "There is a red disk marker on the ground. Your job: guide the robot toward it or stop when close.\n\n"
                "STEP 1 — Find the red disk.\n"
                '- If the red disk is NOT visible, output: {"u": null, "stop": false}\n\n'
                "STEP 2 — Estimate proximity.\n"
                "- NEAR: The red disk appears LARGE and fills a significant portion of the image (roughly the bottom third or more).\n"
                "- FAR: The red disk appears small or distant.\n\n"
                "STEP 3 — Decide action.\n"
                '- If NEAR: output {"u": 0, "stop": true}\n'
                "- If FAR: continue to STEP 4.\n\n"
                "STEP 4 — Measure horizontal offset u (only if FAR).\n"
                "- u = normalized horizontal position of the disk center.\n"
                "- u = -1: disk is at the far left edge.\n"
                "- u = 0: disk is at the image center (straight ahead).\n"
                "- u = +1: disk is at the far right edge.\n"
                "- Use a continuous value, rounded to 2 decimals.\n\n"
                "Output ONLY valid JSON — no explanation, no extra text:\n"
                '{"u": <number -1 to 1 or null>, "stop": <true or false>}\n\n'
                "Examples:\n"
                '- Disk centered and large (NEAR) → {"u": 0, "stop": true}\n'
                '- Disk centered and small (FAR)  → {"u": 0.00, "stop": false}\n'
                '- Disk to the right, small (FAR) → {"u": 0.62, "stop": false}\n'
                '- Disk not visible               → {"u": null, "stop": false}'
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
        if 'heading' in goal:
            self.previous_goal_heading = goal['heading']
        
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
        predictor_type: Type of predictor ('dummy', 'huggingface', 'hf')
        **kwargs: Additional arguments for predictor initialization
            - model_path: Ignored for 'huggingface' (always uses Qwen3-VL-8B-Instruct)
            - device: 'cuda' or 'cpu' (default: 'cuda')
            - trust_remote_code: Trust remote code (default: True)
        
    Returns:
        VLMGoalPredictor instance
        
    Examples:
        # Dummy predictor for testing
        predictor = create_vlm_predictor('dummy')
        
        # Qwen3-VL-8B-Instruct
        predictor = create_vlm_predictor('huggingface', device='cuda')
    """
    if predictor_type == "dummy":
        return DummyVLMPredictor(**kwargs)
    elif predictor_type in ["huggingface", "hf"]:
        return HuggingFaceVLMPredictor(**kwargs)
    elif predictor_type == "openvla":
        return OpenVLAGoalPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}. Choose from: dummy, huggingface")
