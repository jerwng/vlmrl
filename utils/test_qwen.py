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

prompt = """
You are given ONE egocentric RGB image from a quadruped camera.

Task:
Locate the center of the red disk marker and report its horizontal position relative to the image center.

Definition:
- Let u be the normalized horizontal offset in [-1, 1].
- u = -1 means the target center is at the left edge.
- u = 0 means the target center is at the image centerline (forward).
- u = 1 means the target center is at the right edge.

Output ONLY JSON:
{
  "u": <number between -1 and 1>
}

Rules:
- Use a continuous value (not just -1, -0.5, 0, 0.5, 1).
- Round to 2 decimals.
- If the target is not visible, output {"u": nan}.
"""

frame = Image.open(args.image_path)

vlm_goal = vlm_predictor.predict_goal(frame, prompt=prompt, current_heading=current_heading)

print(prompt)
print(vlm_goal)
