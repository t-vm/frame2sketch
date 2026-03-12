import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import config
from PIL import Image
from data import get_transform


def _load_pil(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert('RGB')
    return img


"""    
图像i/o处理类
   
包含图像读取、预处理、显示等功能。
"""
class ImageIO:
    @staticmethod
    def load_pil(image_path: str) -> Image.Image:
        img = Image.open(image_path).convert('RGB')
        return img


    # @staticmethod
    # def read_gray(image_path: str):
    #     img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    #     if img is None:
    #         raise FileNotFoundError(f"cannot read imgae: {image_path}")
    #     if img.ndim == 3:  # color image
    #         if img.shape[2] == 4:
    #             img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # remove alpha channel
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = img.copy()
    #     return gray
    """
    output: tensor;(1, 3, H, W) float
    """
    @staticmethod
    def read_tensor(img_path: str, load_size: int):
        """read tensors from a given image path
        Parameters:
            path (str)     -- input image path
            load_size(int) -- the input size. If <= 0, don't resize
        Returns:
            image_tensor (torch.Tensor) -- the resulting image tensor of shape (1, C, H, W)
            aus_resize (tuple or None)   -- the ORIGINAL image size before resizing, or None if no resizing was done
        """
        img = _load_pil(img_path)
        aus_resize = img.size if load_size > 0 else None
        transform = get_transform(load_size=load_size)
        return transform(img).unsqueeze(0), aus_resize
    @staticmethod
    def to_tensor(pil_img: Image.Image, load_size: int):
        aus_resize = pil_img.size if load_size > 0 else None
        transform = get_transform(load_size=load_size)
        return transform(pil_img).unsqueeze(0), aus_resize


    @staticmethod
    def read_gray(image_path: str):
        img = _load_pil(image_path)
        return np.array(img.convert('L'))  # turn PIL grayscale to numpy uint8(0-255) 
    @staticmethod
    def to_gray(pil_img: Image.Image):
        return np.array(pil_img.convert('L'))


    @staticmethod
    def show_images(images, titles=None, cmap="gray", cols=3, figsize=(14, 8)):
        """show a list of images with matplotlib
        Parameters:
            images (list of np.ndarray) -- LIST of images to show
            titles (list of str)        -- LIST of titles for each image
            cmap (str)                  -- color map for displaying images
            cols (int)                  -- number of columns in the display grid
            figsize (tuple)             -- size of the figure
        """
        n = len(images)
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=figsize)
        for i, img in enumerate(images, 1):
            plt.subplot(rows, cols, i)
            plt.imshow(img, cmap=cmap)
            plt.axis("off")
            if titles:
                plt.title(titles[i-1])
        plt.tight_layout()
        plt.show()


    @staticmethod
    def save_image(image_numpy:np.ndarray, save_path:str, output_resize=None):
        """Save a numpy image to the disk
        Parameters:
            image_numpy (numpy array)    -- input numpy array
            save_path (str)              -- the path to save the image
            output_resize(None or tuple) -- the output size. If None, don't resize)
        """
        image_pil = Image.fromarray(image_numpy)
        if output_resize:
            image_pil = image_pil.resize(output_resize, resample=Image.BICUBIC)
        image_pil.save(save_path)



class ImageOps:
    """
    output:输入灰度图的反相
    """
    @staticmethod
    def invert_gray(img_u8):   
        return 255 - img_u8

    
    # 把任意灰度线稿转成黑线白底。methodology:如果整体偏暗，反相。
    @staticmethod
    def make_blackline_whitebg(gray_line):
        m = gray_line.mean()
        return ImageOps.invert_gray(gray_line) if m < 127 else gray_line
    

    @staticmethod
    def gen_photocopy(
        detail=4,
        gray=None, # pre-read gray image
        img_path_debug=None # for debugging intermediate steps
    ):
        """turn gray image into lineart using a photocopy-like image. 
        Parameters:
            detail (int) -- controls the size of the Gaussian kernel. Recommended range: 3-6.
            gray (np.ndarray) -- pre-read grayscale image. If None, it will be read from img_path_debug.
            img_path_debug (str) -- if provided, the function will read the grayscale image from this path and save intermediate results for debugging.
        Returns:
            result_u8 (np.ndarray) -- the resulting lineart image as a UINT8 NUMPY ARRAY.
        """        
        gray = ImageIO.read_gray(img_path_debug) if img_path_debug  else gray

        # 0~1 normalization
        gray_norm = gray.astype(np.float32) / 255.0

        # blend
        inverted = 1.0 - gray_norm

        # blur
        blur_radius = max(0.6, detail * 0.3)
        kernel_size = int(blur_radius * 2) * 2 + 1
        blurred = cv2.GaussianBlur(inverted, (kernel_size, kernel_size), blur_radius)

        blurred_inverted = 1.0 - blurred

        # color dodge
        dodge = gray_norm / (blurred_inverted + 1e-7)
        dodge = np.clip(dodge, 0, 1)

        # 使得方向都为黑线白底
        result_u8 = ImageOps.make_blackline_whitebg(dodge * 255).astype(np.uint8)

        return result_u8
    

    def blend_lines_np(structure_line, style_line, alpha=0.6, beta=0.4, mask_blur=3):
        """blend structure line and style line using weighted addition. 
        Parameters:
            structure_line (np.ndarray) -- the structure line image as a uint8 numpy array (H, W)
            style_line (np.ndarray)     -- the style line image as a uint8 numpy array (H, W, C)
            alpha (float)              -- weight for structure line
            beta (float)               -- weight for style line
            mask_blur (int)            -- kernel size for blurring the structure line mask  
        Returns:
            blended (np.ndarray)       -- the blended lineart image as a uint8 numpy array (H, W, C)
        """
        # resize
        h, w = structure_line.shape[:2]
        if style_line.shape[:2] != (h, w):
            style_line = cv2.resize(style_line, (w, h), interpolation=cv2.INTER_LINEAR)

        # 生成结构线的mask
        mask = cv2.GaussianBlur(structure_line, (mask_blur, mask_blur), 0)
        mask = mask.astype(np.float32) / 255.0

        # print(structure_line.shape, style_line.shape, mask.shape)  # debug
        # 融合线稿
        blended = (alpha * structure_line[:, :, np.newaxis] + beta * style_line) * mask[:, :, np.newaxis]
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return blended


    def adaptive_darken(
        img: np.ndarray,
        gamma: float = 3.1,  # gamma > 1 will darken the image
        line_threshold:float = 240,
        blackenough_line_threshold:float = 210
    ) -> np.ndarray:
        """Automatically darken the image if the lines are too light.
        Parameters:
            img (np.ndarray) -- uint8 numpy array (H, W, C) or (H, W)
            gamma (float) -- the gamma value for darkening; values greater than 1 will darken the image.
            line_threshold (float) -- the threshold for determining if the lines are too light. If the average pixel value of the LINES is above this threshold, img will be darkened.
        """
        lines_mask = img <line_threshold
        if np.mean(img[lines_mask]) < blackenough_line_threshold:
            return img  # lines already dark enough
        
        lut = (np.arange(256) / 255.0) ** gamma * 255
        result = cv2.LUT(img, lut.astype(np.uint8))
        return result