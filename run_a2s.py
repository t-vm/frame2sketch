"""Test script for anime-to-sketch translation
Example:
    python3 test.py --dataroot /your_path/dir --load_size 512
    python3 test.py --dataroot /your_path/img.jpg --load_size 512
"""


import os
import torch
from tqdm.auto import tqdm
from kornia.enhance import equalize_clahe

from config import TestConfig
from data import get_image_list, read_img_path, tensor_to_img, save_image
from model import create_model


def prepare_device(gpu_ids):
    gpu_list = ','.join(str(x) for x in gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    return torch.device('cuda' if len(gpu_ids) > 0 else 'cpu')


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
    cfg = TestConfig()

    device = prepare_device(cfg.GPU_IDS)
    model = create_model(cfg.MODEL).to(device)
    model.eval()

    test_list = get_test_list(cfg.DATAROOT)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    for test_path in tqdm(test_list):
        basename = os.path.basename(test_path)
        aus_path = os.path.join(cfg.OUTPUT_DIR, basename)

        img, aus_resize = read_img_path(test_path, cfg.LOAD_SIZE)
        img = preprocess_image(img, cfg.CLAHE_CLIP)

        with torch.no_grad():
            aus_tensor = model(img.to(device))

        aus_img = tensor_to_img(aus_tensor)
        save_image(aus_img, aus_path, aus_resize)


if __name__ == '__main__':
    main()