import torch
import torch.nn as nn
import argparse
import random
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Inspect a single shard of the dataset")
    parser.add_argument("--shard_path", type=str, required=True, help="Path to the .pt shard file")
    parser.add_argument("--output_plot", type=str, default="inspect_shard.png", help="Path to save the inspection plot")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to inspect")
    
    args = parser.parse_args()

    # Load shard
    print(f"Loading shard from {args.shard_path}...")
    try:
        data = torch.load(args.shard_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading shard: {e}")
        return

    print(f"Shard loaded. Total samples: {len(data)}")
    
    if len(data) == 0:
        print("Shard is empty.")
        return

    # Sample random videos
    if len(data) < args.num_samples:
        print(f"Warning: Shard has fewer samples ({len(data)}) than requested ({args.num_samples}). Showing all.")
        samples = data
    else:
        samples = random.sample(data, args.num_samples)

    # Visualize
    fig, axes = plt.subplots(len(samples), 2, figsize=(20, 5 * len(samples)))
    
    # Handle case where num_samples is 1 (axes is 1D array)
    if len(samples) == 1:
        axes = [axes]

    for i, sample in enumerate(samples):
        video_id = sample.get('video_id', 'Unknown ID')
        fbank = sample['fbank']
        images_bytes = sample['images']
        
        # 1. Plot Fbank
        ax_fbank = axes[i][0]
        # fbank is [Time, Mel], transpose for plotting -> [Mel, Time]
        # Ensure it's on CPU and float32 for plotting
        if isinstance(fbank, torch.Tensor):
            fbank_np = fbank.float().cpu().t().numpy()
        else:
            fbank_np = fbank.T.astype(np.float32)
            
        ax_fbank.imshow(fbank_np, aspect='auto', origin='lower', cmap='inferno')
        ax_fbank.set_title(f"Log Mel Spectrogram - {video_id}")
        ax_fbank.set_xlabel("Time Frames")
        ax_fbank.set_ylabel("Mel Bins")

        # 2. Plot Frames (Grid of 16 images)
        ax_frames = axes[i][1]
        ax_frames.axis('off')
        
        # Create a sub-grid for the 16 frames within the frames axis
        # We can't easily do subplots inside a subplot axe, so we'll stitch images together
        
        frames = []
        for img_byte in images_bytes:
            img = Image.open(io.BytesIO(img_byte))
            frames.append(np.array(img))
        
        # Assuming 16 frames, 4x4 grid
        # If not 16, try to make a square grid
        num_frames = len(frames)
        grid_size = int(np.ceil(np.sqrt(num_frames)))
        
        if num_frames > 0:
            frame_h, frame_w, _ = frames[0].shape
            
            # Create canvas
            grid_h = grid_size * frame_h
            grid_w = grid_size * frame_w
            canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            
            for idx, frame in enumerate(frames):
                r = idx // grid_size
                c = idx % grid_size
                y_start = r * frame_h
                x_start = c * frame_w
                canvas[y_start:y_start+frame_h, x_start:x_start+frame_w] = frame
            
            ax_frames.imshow(canvas)
            ax_frames.set_title(f"Extracted Frames ({num_frames})")
        else:
            ax_frames.text(0.5, 0.5, "No frames found", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Inspection plot saved to {args.output_plot}")

if __name__ == "__main__":
    main()
