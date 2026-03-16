from vlm.vlm_goal_predictor import create_vlm_predictor
import argparse
from PIL import Image
    
parser = argparse.ArgumentParser(description='Run Go2 with Qwen3-VL goal prediction')

parser.add_argument('--vlm_device', type=str, default='cuda',
                    help='Device for VLM model: "cuda", "cpu", or "cuda:0", "cuda:1", etc. (default: cuda)')

parser.add_argument('--trust_remote_code', action='store_true', default=True,
                    help='Trust remote code when loading model (default: True)')

parser.add_argument('--image_path', type=str, required=True,
                    help='Path to the input image file')

args = parser.parse_args()


vlm_predictor = create_vlm_predictor(
    'huggingface',
    device=args.vlm_device,
    trust_remote_code=args.trust_remote_code
)

current_heading = 0.0  # Example current heading in radians

prompt = """You are controlling a robot navigating toward a red disk target.

Detect the red disk target in the image and return ONLY valid JSON:
{"bbox": [x1, y1, x2, y2], "label": "object name"}

Rules:
- Coordinates are in 0-1000 scale relative to image dimensions
- Tightly bound the VISIBLE portion of the object
- Detect even if partially occluded or cut off by image edge
- If target not visible: {"bbox": null, "label": null}

Target: {target}
"""

frame = Image.open(args.image_path)

vlm_goal = vlm_predictor.predict_goal(frame, prompt=prompt, current_heading=current_heading)

print(prompt)
print(vlm_goal)
