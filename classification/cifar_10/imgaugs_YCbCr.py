
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, ColorJitter
from torchvision.transforms.functional import _is_pil_image


def convert_colorspace(img, mode):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if mode not in ("RGB", "YCbCr", "HSV"):
        raise TypeError('mode should be one of "RGB", "YCbCr", "HSV". Got {}'.format(mode))
    return img.convert(mode)


class ConvertColorspace(object):
    def __init__(self, mode):

        assert mode in ("RGB", "YCbCr", "HSV")
        self.mode = mode

    def __call__(self, img):
        return convert_colorspace(img, self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '(mode={})'.format(self.mode)


train_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(hue=0.1, brightness=0.1),
    ConvertColorspace("YCbCr"),
]


val_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ConvertColorspace("YCbCr"),
]


test_imgaugs = val_imgaugs
