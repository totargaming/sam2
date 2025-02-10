import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2_video_predictor
import gc
import time
import json
from skimage.measure import find_contours

# Set device for computation
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print("\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might give numerically different outputs and sometimes degraded performance on MPS. See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion.")

# Initialize predictor
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Input and output directories
frame_dir = "./datasets/Skincare"
output_dir = "./datasets/Skincare_output"
os.makedirs(output_dir, exist_ok=True)

# Process video frames
frame_names = sorted([p for p in os.listdir(frame_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]], key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))
inference_state = predictor.init_state(video_path=frame_dir)
predictor.reset_state(inference_state)

points, labels = np.array([[510,250]], dtype=np.float32), np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=0, obj_id=1, points=points, labels=labels)

video_segments = {out_frame_idx: {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)} for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state)}

# Start timing
start_time = time.time()

mask_data = {}

def mask_to_polygon(mask):
    # Convert mask to binary
    mask_binary = mask.astype(bool)
    print(f"mask shape: {mask.shape}, mask_binary shape: {mask_binary.shape}")
    print(f"mask_binary unique values: {np.unique(mask_binary)}")
    
    # Check if mask is empty
    if mask_binary.size == 0 or np.all(mask_binary == 0):
        return []
    
    # Ensure mask is 2D
    if mask_binary.ndim > 2:
        mask_binary = mask_binary[0]
    
    # Find contours
    contours = find_contours(mask_binary, level=0.5)
    # Convert contours to polygons
    polygons = [contour.tolist() for contour in contours]
    return polygons

for frame_idx in range(len(frame_names)):
    frame_masks = []
    for obj_id, mask in video_segments.get(frame_idx, {}).items():
        polygons = mask_to_polygon(mask)
        frame_masks.append({
            "obj_id": obj_id,
            "polygons": polygons
        })
    mask_data[frame_idx] = frame_masks
    print(f"Processed frame {frame_idx + 1}/{len(frame_names)}")

# Save mask data to JSON
with open(os.path.join(output_dir, "mask_data.json"), "w") as json_file:
    json.dump(mask_data, json_file, indent=4)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")