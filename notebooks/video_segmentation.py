import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2_video_predictor

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

def show_points(coords, labels, ax, marker_size=200):
    ax.scatter(*coords[labels==1].T, color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(*coords[labels==0].T, color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    ax.add_patch(plt.Rectangle(box[:2], *(box[2:] - box[:2]), edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Process video frames
video_dir = "./videos/bedroom"
frame_names = sorted([p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]], key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

points, labels = np.array([[210, 350], [250, 220]], dtype=np.float32), np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(inference_state, frame_idx=0, obj_id=1, points=points, labels=labels)

video_segments = {out_frame_idx: {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)} for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state)}

def save_frame_as_image(frame_idx, frame_names, video_dir, video_segments, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    ax.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
    for obj_id, mask in video_segments.get(frame_idx, {}).items():
        show_mask(mask, ax, obj_id=obj_id)
    plt.savefig(os.path.join(output_dir, f"frame_{frame_idx:04d}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)
for out_frame_idx in range(len(frame_names)):
    save_frame_as_image(out_frame_idx, frame_names, video_dir, video_segments, output_dir)

def create_video_from_frames(output_dir, output_video_path, fps=30):
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
    frame = cv2.imread(os.path.join(output_dir, frame_files[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_file in frame_files:
        video.write(cv2.imread(os.path.join(output_dir, frame_file)))
    video.release()

create_video_from_frames(output_dir, "output_video.mp4")