import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2_video_predictor
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc
import time

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

def show_mask(mask, ax, obj_id=None, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([*plt.get_cmap("tab10")(0 if obj_id is None else obj_id)[:3], 0.6])
    ax.imshow(mask.reshape(*mask.shape[-2:], 1) * color.reshape(1, 1, -1))

def show_box(box, ax):
    ax.add_patch(plt.Rectangle(box[:2], *(box[2:] - box[:2]), edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Input and output directories
frame_dir = "./datasets/flour"
output_dir = "./datasets/flour_outputbox"
os.makedirs(output_dir, exist_ok=True)

# Process video frames
frame_names = sorted([p for p in os.listdir(frame_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]], key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))
inference_state = predictor.init_state(video_path=frame_dir)
predictor.reset_state(inference_state)

# Add a box to get started
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
box = np.array([380, 360, 490, 435], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, box=box)

video_segments = {out_frame_idx: {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)} for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state)}

# Start timing
start_time = time.time()

for frame_idx in range(len(frame_names)):
    frame = Image.open(os.path.join(frame_dir, frame_names[frame_idx]))
    fig, ax = plt.subplots(figsize=(frame.width / 100, frame.height / 100), dpi=100)
    ax.axis('off')
    ax.imshow(frame)
    for obj_id, mask in video_segments.get(frame_idx, {}).items():
        show_mask(mask, ax, obj_id=obj_id)
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_argb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, [1, 2, 3]]  # Convert ARGB to RGB
    output_frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
    cv2.imwrite(output_frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.close(fig)
    del frame, fig, ax, canvas, img  # Free up memory
    gc.collect()  # Force garbage collection
    print(f"Processed frame {frame_idx + 1}/{len(frame_names)}")

# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")