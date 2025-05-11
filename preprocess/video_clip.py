import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_fixed_frames(video_path, num_frames=128):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)
    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames

def preprocess_frames(frames, processor):
    pil_frames = [Image.fromarray(frame) for frame in frames]
    return processor(images=pil_frames, return_tensors="pt", padding=True).to(device)

def extract_clip_features(frames, model, processor):
    processed_frames = preprocess_frames(frames, processor)
    with torch.no_grad():
        features = model.get_image_features(**processed_frames)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features

def pool_video_features(features, pool_type="mean"):
    if pool_type == "mean":
        return features.mean(dim=0)
    elif pool_type == "max":
        return features.max(dim=0).values
    else:
        raise ValueError("Unsupported pool_type. Use 'mean' or 'max'.")

def extract_and_save_video_feature(video_path, model, processor, output_file, num_frames=128, pool_type="mean"):
    frames = extract_fixed_frames(video_path, num_frames)
    features = extract_clip_features(frames, model, processor)
    video_feature = pool_video_features(features, pool_type=pool_type)
    np.save(output_file, video_feature.cpu().numpy())
    print(f"save video feature as {output_file}")

def extract_features_from_directory(input_dir, output_dir, model, processor, num_frames=128, pool_type="mean", max_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".mp4")]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_and_save_video_feature, video_path, model, processor,
                os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.npy"),
                num_frames, pool_type
            ): video_path for video_path in video_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Videos"):
            video_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file {video_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video features using CLIP")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .mp4 videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npy feature files")
    parser.add_argument("--num_frames", type=int, default=128, help="Number of frames to sample from each video")
    parser.add_argument("--pool_type", type=str, choices=["mean", "max"], default="mean", help="Pooling method")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of threads for parallel processing")
    
    args = parser.parse_args()

    extract_features_from_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=model,
        processor=processor,
        num_frames=args.num_frames,
        pool_type=args.pool_type,
        max_workers=args.max_workers
    )
