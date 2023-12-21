from pathlib import Path
from typing import cast
from typing import List, Optional, Union
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import math

#----------------------------------------------------------------------------
here = Path(__file__).parent.absolute()
comfy_dir = here.parent.parent

#----------------------------------------------------------------------------
def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)

    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

#----------------------------------------------------------------------------
# 添加一个辅助函数，用于交换宽度和高度
def swap_width_height(width, height):
    return height, width

#----------------------------------------------------------------------------
import logging
import re
import os

base_log_level = logging.DEBUG if os.environ.get("MTB_DEBUG") else logging.INFO


# Custom object that discards the output
class NullWriter:
    def write(self, text):
        pass


class Formatter(logging.Formatter):
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    purple = "\x1b[35;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "[%(name)s] | %(levelname)s -> %(message)s"

    FORMATS = {
        logging.DEBUG: purple + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def mklog(name, level=base_log_level):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(Formatter())
    logger.addHandler(ch)

    # Disable log propagation
    logger.propagate = False

    return logger


# - The main app logger
log = mklog(__package__, base_log_level)


def log_user(arg):
    print("\033[34mComfy MTB Utils:\033[0m {arg}")


def get_summary(docstring):
    return docstring.strip().split("\n\n", 1)[0]


def blue_text(text):
    return f"\033[94m{text}\033[0m"


def cyan_text(text):
    return f"\033[96m{text}\033[0m"


def get_label(label):
    words = re.findall(r"(?:^|[A-Z])[a-z]*", label)
    return " ".join(words).strip()

# 禁用 aiohttp 的访问日志记录器
logging.getLogger('aiohttp.access').disabled = True
#----------------------------------------------------------------------------
def bbox_dim(bbox):
    left, upper, right, lower = bbox
    width = right - left
    height = lower - upper
    return width, height

#----------------------------------------------------------------------------
class Text_Image_Zho:

    fonts = {}

    def __init__(self):
        # - This is executed when the graph is executed, we could conditionaly reload fonts there
        pass

    @classmethod
    def CACHE_FONTS(cls):
        font_extensions = ["*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot"]
        fonts = []

        for extension in font_extensions:
            fonts.extend(comfy_dir.glob(f"**/{extension}"))

        if not fonts:
            log.warn(
                "> No fonts found in the comfy folder, place at least one font file somewhere in ComfyUI's hierarchy"
            )
        else:
            log.debug(f"> Found {len(fonts)} fonts")

        for font in fonts:
            log.debug(f"Adding font {font}")
            cls.fonts[font.stem] = font.as_posix()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.fonts:
            cls.CACHE_FONTS()
        else:
            log.debug(f"Using cached fonts (count: {len(cls.fonts)})")
        return {
            "required": {
                "text": (
                    "STRING",
                        {"default": "ZHOZHOZHO"}, 
                ),
                "selected_font": ((sorted(cls.fonts.keys())),),
                "align": (["left", "center", "right"],
                ),
                "wrap": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "font_size": (
                    "INT",
                    {"default": 12, "min": 1, "max": 2500, "step": 1},
                ),
                "color": (
                    "COLOR",
                    {"default": "red"},
                ),
                "outline_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "outline_color": (
                    "COLOR",
                    {"default": "blue"},  # 设置默认的描边颜色
                ),
                "margin_x": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "margin_y": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8096, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8096, "step": 1},
                ),
                "swap": ("BOOLEAN", {"default": False}),  # 添加交换宽度和高度的按钮
                "arc_text": ("BOOLEAN", {"default": False}),
                "arc_radius": (
                    "INT",
                    {"default": 100, "min": 1, "max": 2500, "step": 1},
                ),
                "arc_start_angle": (
                    "INT",
                    {"default": 180, "min": 0, "max": 360, "step": 1},
                ),
                "arc_end_angle": (
                    "INT",
                    {"default": 360, "min": 0, "max": 360, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "text_to_image"
    CATEGORY = "Zho模块组/text"
    
    def draw_text_in_arc(self, image, draw, text, font, font_path, font_size, center, radius, start_angle, end_angle, fill='black', stroke_fill='blue', stroke_width=0):
        # 文字的总长度
        text_width = sum(font.getsize(char)[0] for char in text[:-1])

        # 根据扇形角度计算角度步长
        angle_range = end_angle - start_angle
        angle_step = (angle_range / (len(text) - 1) if len(text) > 1 else 1)

        # 开始绘制文字
        current_angle = start_angle
        for char in text:
            char_width, char_height = font.getsize(char)
            angle = math.radians(current_angle)
            print(current_angle, angle, angle_step)

            # 创建单独的图像用于旋转字符
            super_sampling_multiplier = 10
            char_image = Image.new("RGBA", (char_width * super_sampling_multiplier, char_height * super_sampling_multiplier), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_image)
            super_sampling_font = cast(ImageFont.FreeTypeFont, ImageFont.truetype(font_path, font_size * super_sampling_multiplier))
            char_draw.text((0, 0), char, font=super_sampling_font, fill=fill, stroke_fill=stroke_fill, stroke_width=stroke_width)

            # 计算旋转角度和字符位置
            rotate_angle = current_angle - 90 - (current_angle - 180) * 2 # 使字符面向圆心
            rotated_char_image = char_image.rotate(rotate_angle, expand=1, resample=Image.BICUBIC)
            # 缩小图像大小
            new_size = (int(rotated_char_image.width / 10), int(rotated_char_image.height / 10))
            rotated_char_image_resized = rotated_char_image.resize(new_size, Image.ANTIALIAS)

            # 计算缩小后的图像放置位置
            x = center[0] + radius * math.cos(angle) - rotated_char_image_resized.size[0] / 2
            y = center[1] + radius * math.sin(angle) - rotated_char_image_resized.size[1] / 2

            # 粘贴缩小后的图像
            image.paste(rotated_char_image_resized, (int(x), int(y)), rotated_char_image_resized)

            # 更新下一个字符的开始角度
            current_angle += angle_step


    def text_to_image(
        self, text, selected_font, align, wrap, font_size, width, height, color, outline_size, outline_color, margin_x, margin_y, swap=False, arc_text=False, arc_radius=100, arc_start_angle=180, arc_end_angle=360
    ):
        import textwrap

        # 如果用户选择交换宽度和高度，则调用交换函数
        if swap:
            width, height = swap_width_height(width, height)

        font_path = self.fonts[selected_font]
        (_, top, _, _) = ImageFont.truetype(font_path, font_size).getbbox(text)
        font = cast(ImageFont.FreeTypeFont, ImageFont.truetype(font_path, font_size))
        if wrap == 0:
            wrap = width / font_size
        wrap = int(wrap)
        lines = textwrap.wrap(text, width=wrap)
        log.debug(f"Lines: {lines}")
        line_height = bbox_dim(font.getbbox("hg"))[1]
        img_height = height  # line_height * len(lines)
        img_width = width  # max(font.getsize(line)[0] for line in lines)

        img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 曲线文字
        if arc_text:
            width, height = bbox_dim(font.getbbox(text))
            
            center_x = (img_width) // 2
            center_y = arc_radius + (height)
            
            if align == "left":
                center_x = arc_radius + (height) // 2
            elif align == "right":
                center_x = img_width - arc_radius - (height) // 2
            center = (center_x + margin_x, center_y + margin_y)
            self.draw_text_in_arc(img, draw, text, font, font_path, font_size, center, arc_radius, arc_start_angle, arc_end_angle,
                                  fill=color, stroke_fill=outline_color, stroke_width=outline_size)
        else:
            # 初始化 y_text
            y_text = margin_y + outline_size - top
        
            for line in lines:
                width, height = bbox_dim(font.getbbox(line))

                # 根据 align 参数计算文本的 x 坐标
                if align == "left":
                    x_text = margin_x
                elif align == "center":
                    x_text = (img_width - width) // 2
                elif align == "right":
                    x_text = img_width - width - margin_x
                else:
                    x_text = margin_x  # 默认为左对齐

                draw.text(
                    (x_text, y_text),
                    text=line,
                    fill=color,
                    stroke_fill=outline_color,
                    stroke_width=outline_size,
                    font=font,
                )
                y_text += height

        return (pil2tensor(img),)

#----------------------------------------------------------------------------
class Text_Image_Multiline_Zho:

    fonts = {}

    def __init__(self):
        # - This is executed when the graph is executed, we could conditionaly reload fonts there
        pass

    @classmethod
    def CACHE_FONTS(cls):
        font_extensions = ["*.ttf", "*.otf", "*.woff", "*.woff2", "*.eot"]
        fonts = []

        for extension in font_extensions:
            fonts.extend(comfy_dir.glob(f"**/{extension}"))

        if not fonts:
            log.warn(
                "> No fonts found in the comfy folder, place at least one font file somewhere in ComfyUI's hierarchy"
            )
        else:
            log.debug(f"> Found {len(fonts)} fonts")

        for font in fonts:
            log.debug(f"Adding font {font}")
            cls.fonts[font.stem] = font.as_posix()

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.fonts:
            cls.CACHE_FONTS()
        else:
            log.debug(f"Using cached fonts (count: {len(cls.fonts)})")
        return {
            "required": {
                "text": (
                    "STRING",
                        {"default": "ZHOZHOZHO", "multiline": True}, 
                ),
                "selected_font": ((sorted(cls.fonts.keys())),),
                "align": (["left", "center", "right"],
                ),
                "wrap": (
                    "INT",
                    {"default": 120, "min": 0, "max": 8096, "step": 1},
                ),
                "graphspace": (
                    "INT",
                    {"default": 10, "min": 0, "max": 8096, "step": 1},
                ),
                "linespace": (
                    "INT",
                    {"default": 2, "min": 0, "max": 8096, "step": 1},
                ),
                "font_size": (
                    "INT",
                    {"default": 12, "min": 1, "max": 2500, "step": 1},
                ),
                "color": (
                    "COLOR",
                    {"default": "red"},
                ),
                "outline_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "outline_color": (
                    "COLOR",
                    {"default": "blue"},  # 设置默认的描边颜色
                ),
                "margin_x": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "margin_y": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8096, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8096, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8096, "step": 1},
                ),
                "swap": ("BOOLEAN", {"default": False}),  # 添加交换宽度和高度的按钮
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "text_to_image_multiline"
    CATEGORY = "Zho模块组/text"

    def text_to_image_multiline(
        self, text, selected_font, align, wrap, graphspace, linespace, font_size, width, height, color, outline_size, outline_color, margin_x, margin_y, swap=False
    ):
        from PIL import Image, ImageDraw, ImageFont
        import textwrap

        # 如果用户选择交换宽度和高度，则调用交换函数
        if swap:
            width, height = swap_width_height(width, height)

        font_path = self.fonts[selected_font]
        (_, top, _, _) = ImageFont.truetype(font_path, font_size).getbbox(text)
        font = cast(ImageFont.FreeTypeFont, ImageFont.truetype(font_path, font_size))
        if wrap == 0:
            wrap = width / font_size
        wrap = int(wrap)

        paragraphs = text.split('\n')

        log.debug(f"Paragraphs: {paragraphs}")

        img_height = height  # line_height * len(lines)
        img_width = width  # max(font.getsize(line)[0] for line in lines)

        img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 初始化 y_text
        y_text = margin_y + outline_size

        for paragraph in paragraphs:
            lines = textwrap.wrap(paragraph, width=wrap, expand_tabs=False, replace_whitespace=False)

            for line in lines:
                width, height = bbox_dim(font.getbbox(line))

               # 根据 align 参数重新计算 x 坐标
                if align == "left":
                    x_text = margin_x
                elif align == "center":
                    x_text = (img_width - width) // 2
                elif align == "right":
                    x_text = img_width - width - margin_x
                else:
                    x_text = margin_x  # 默认为左对齐

                draw.text(
                    (x_text, y_text),
                    text=line,
                    fill=color,
                    stroke_fill=outline_color,
                    stroke_width=outline_size,
                    font=font,
                )

                # 更新 y 坐标，加上当前行的高度和一些额外的间距
                y_text += height + linespace  # linespace 是行之间的额外间距

            # 段落之间添加一些额外的间距
            y_text += graphspace  # 可以根据需要调整

        return (pil2tensor(img),)









#----------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "Text_Image_Zho": Text_Image_Zho,
    "Text_Image_Multiline_Zho": Text_Image_Multiline_Zho,
}