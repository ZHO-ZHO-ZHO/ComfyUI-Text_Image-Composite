import torch

class AlphaChanelAddByMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "method": (["default", "invert"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = "Zho模块组/image"

    def node(self, images, mask, method):
        img_height, img_width = images[0, :, :, 0].shape
        mask_height, mask_width = mask.shape

        if img_height != mask_height or img_width != mask_width:
            raise ValueError(
                "[AlphaChanelByMask]: Size of images not equals size of mask. " +
                "Images: [" + str(img_width) + ", " + str(img_height) + "] - " +
                "Mask: [" + str(mask_width) + ", " + str(mask_height) + "]."
            )

        if method == "default":
            return (torch.stack([
                torch.stack((
                    images[i, :, :, 0],
                    images[i, :, :, 1],
                    images[i, :, :, 2],
                    1. - mask
                ), dim=-1) for i in range(len(images))
            ]),)
        else:
            return (torch.stack([
                torch.stack((
                    images[i, :, :, 0],
                    images[i, :, :, 1],
                    images[i, :, :, 2],
                    mask
                ), dim=-1) for i in range(len(images))
            ]),)


NODE_CLASS_MAPPINGS = {
    "AlphaChanelAddByMask": AlphaChanelAddByMask,
}
