from PIL import Image, ImageOps
import torchvision.transforms as transforms

class ResizeWithPadding(object):
    def __init__(self, size, fill_color=(0, 0, 0)):
        assert isinstance(size, (tuple, list)) and len(size) == 2
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img):
        img.thumbnail(self.size, Image.Resampling.LANCZOS)
        delta_w = self.size[0] - img.size[0]
        delta_h = self.size[1] - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(img, padding, fill=self.fill_color)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, fill_color={1})'.format(self.size, self.fill_color)