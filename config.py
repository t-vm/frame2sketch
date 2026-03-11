from pathlib import Path


class AnimeToSketchConfig:
    def __init__(self):
        self.DATAROOT = 'test_samples/'
        self.LOAD_SIZE = 512
        self.MODEL_DIR = 'weights/netG.pth'
        self.IMPROVED_MODEL_DIR = 'weights/improved.bin'
        self.OUTPUT_DIR = 'results/'
        self.GPU_IDS = []
        self.MODEL = 'default'
        self.CLAHE_CLIP = -1


class LineartConfig:
    def __init__(self):
        self.IMG_PATH = "./test_samples/imas.png"  # original image path
        self.a2s_out_path = "imas_a2s.png"   # anime2sketch输出路径（若已生成）

        self.OUT_DIR = Path("lineart_outputs")
        self.OUT_DIR.mkdir(exist_ok=True)
        #Photocopy args
        self.DETAIL = 4  #  detail (int) -- controls the size of the Gaussian kernel. Recommended range: 3-6
        self.BLEND_ALPHA = 0.72       # structure透明度
        self.BLEND_BETA = 0.28        # style透明度 
        self.MASK_BLUR = 3            # 结构mask轻微平滑