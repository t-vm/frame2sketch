"""Video to Sketch script for anime-to-sketch translation
Example:
    python3 video_to_sketch.py --input_dir input_video --output_dir results_frames
"""
import os
import torch
from tqdm.auto import tqdm
from kornia.enhance import equalize_clahe
import cv2
import numpy as np
from PIL import Image

from config import AnimeToSketchConfig, LineartConfig
from utils import ImageIO, ImageOps, Utils
from data import tensor_to_img, save_image
from model import create_model


def main(input_dir, output_dir):
    cfg = AnimeToSketchConfig()
    lineart_cfg = LineartConfig()

    device = Utils.prepare_device(cfg.GPU_IDS)
    model = create_model(cfg.MODEL).to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Get list of video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in video_extensions]

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB and to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

           
            # 1.5. preprocess for anime2sketch
            img_np, aus_resize = ImageIO.to_tensor(pil_img, cfg.LOAD_SIZE)
            img_np = ImageIO.preprocess_image(img_np, cfg.CLAHE_CLIP)
            # 2. inference
            with torch.no_grad():
                aus_tensor = model(img_np.to(device))
            img_style_np = tensor_to_img(aus_tensor) # (H, W, C) uint8

            # # debug
            # ksize=(5,5)
            # sigma=1.5
            # img_style_np = cv2.GaussianBlur(img_style_np.astype(np.float32), ksize, sigma).clip(0,255).astype(img_style_np.dtype)
        
        
            # 3. gen lineart layer 
            grey_img = ImageIO.to_gray(pil_img)
            # 生成结构线
            img_structure_np = ImageOps.gen_photocopy(
                detail=lineart_cfg.DETAIL,
                gray=grey_img
            )  # np.array uint8
        
            # 4.sketch fusion 
            blended = ImageOps.blend_lines_np(img_structure_np, img_style_np, alpha=lineart_cfg.BLEND_ALPHA, beta=lineart_cfg.BLEND_BETA, mask_blur=lineart_cfg.MASK_BLUR)  # blend_alpha means the weight for structure line, blend_beta means the weight for style line
            fin_np = ImageOps.adaptive_darken(blended)  # 自动判断是否后处理增强，针对线条较淡的情况        
            # fin_np = blended  # debug

            # Save sketch
            frame_filename = f"frame_{frame_count:06d}.png"
            frame_path = os.path.join(video_output_dir, frame_filename)
            save_image(fin_np, frame_path, aus_resize)


            frame_count += 1

        cap.release()
        print(f"Processed {frame_count} frames for {video_name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert videos to sketches")
    parser.add_argument('--input_dir', type=str, default='input_video', help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, default='results_frames', help='Directory to save output frames')
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)