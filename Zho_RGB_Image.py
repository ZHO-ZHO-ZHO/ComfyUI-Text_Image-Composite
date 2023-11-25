import torch
from PIL import Image
from typing import List, Optional, Union
import numpy as np

def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# 添加一个辅助函数，用于交换宽度和高度
def swap_width_height(width, height):
    return height, width

class RGB_Image_Zho:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": 8160}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8160}),
                "swap": ("BOOLEAN", {"default": False}),  # 添加交换宽度和高度的按钮
                "color": ("COLOR",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rgb_image"
    CATEGORY = "Zho模块组/image"

    def rgb_image(self, color, width, height, swap=False):
        # 如果用户选择交换宽度和高度，则调用交换函数
        if swap:
            width, height = swap_width_height(width, height)

        # 创建RGBA图像
        image = Image.new("RGB", (width, height), color=color)

        # 转换为张量
        image = pil2tensor(image)

        return (image,)


#----------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "RGB_Image_Zho": RGB_Image_Zho,
}