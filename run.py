"""Test script for anime-to-sketch translation
Example:
    python3 test.py --dataroot /your_path/dir --load_size 512
    python3 test.py --dataroot /your_path/img.jpg --load_size 512
"""
import os
import torch
from tqdm.auto import tqdm
from kornia.enhance import equalize_clahe

from config import AnimeToSketchConfig, LineartConfig
from utils import ImageIO, ImageOps

from data import get_image_list, read_img_path, tensor_to_img, save_image
from model import create_model


import numpy as np
import cv2



def prepare_device(gpu_ids: list):
    # gpu_list = ','.join(str(x) for x in gpu_ids)
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    if torch.cuda.is_available():
        print("use device: cuda")
        return torch.device('cuda')
    elif torch.backend.mps.is_available():
        print("use device: mps")
        return torch.device("mps")
    print("use device: cpu")
    return torch.device('cpu')


def get_test_list(dataroot):
    if os.path.isdir(dataroot):
        return get_image_list(dataroot)
    if os.path.isfile(dataroot):
        return [dataroot]
    raise ValueError(f'{dataroot} is not a valid directory or image file.')


def preprocess_image(img, clahe_clip):
    if clahe_clip > 0:
        img = (img + 1) / 2
        img = equalize_clahe(img, clip_limit=clahe_clip)
        img = (img - 0.5) / 0.5
    return img


def main():
    cfg = AnimeToSketchConfig()
    lineart_cfg = LineartConfig()

    device = prepare_device(cfg.GPU_IDS)
    model = create_model(cfg.MODEL).to(device)
    model.eval()

    test_list = get_test_list(cfg.DATAROOT)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    for test_path in tqdm(test_list):
        basename = os.path.basename(test_path)
        aus_path = os.path.join(cfg.OUTPUT_DIR, basename)
        
        # 1. read img 
        pil_img = ImageIO.load_pil(test_path)

        # 1.5. preprocess for anime2sketch
        img_np, aus_resize = ImageIO.to_tensor(pil_img, cfg.LOAD_SIZE)
        img_np = preprocess_image(img_np, cfg.CLAHE_CLIP)
        # 2. inference
        with torch.no_grad():
            aus_tensor = model(img_np.to(device))
        img_style_np = tensor_to_img(aus_tensor) # (H, W, C) uint8

        # debug
        ksize=(5,5)
        sigma=1.5
        img_style_np = cv2.GaussianBlur(img_style_np.astype(np.float32), ksize, sigma).clip(0,255).astype(img_style_np.dtype)
        
        
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
        save_image(fin_np, aus_path, aus_resize)


if __name__ == '__main__':
    main()