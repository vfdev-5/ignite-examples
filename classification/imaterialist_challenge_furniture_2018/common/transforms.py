import random
import numpy as np

from PIL import Image

from torch.utils.data import Dataset


def basic_random_blend(img, img_to_paste, alpha=0.5):
    size = (min(img.size[0], img_to_paste.size[0]), min(img.size[1], img_to_paste.size[1]))
    x = random.randint(0, img_to_paste.size[0] - size[0])
    y = random.randint(0, img_to_paste.size[1] - size[1])
    fg_img = img_to_paste.crop((x, y, x + size[0], y + size[1]))
    x = random.randint(0, img.size[0] - size[0])
    y = random.randint(0, img.size[1] - size[1])
    bg_img = img.crop((x, y, x + size[0], y + size[1]))
    blend_img = Image.blend(bg_img, fg_img, alpha)
    return blend_img


def basic_random_half_blend(img, img_to_paste, alpha=0.3):
    size = (min(img.size[0], img_to_paste.size[0]), min(img.size[1], img_to_paste.size[1]))
    nsize = list(size)
    axis = random.randint(0, 1)
    nsize[axis] = nsize[axis] // 2
    x = random.randint(0, img_to_paste.size[0] - nsize[0])
    y = random.randint(0, img_to_paste.size[1] - nsize[1])
    fg_img = img_to_paste.crop((x, y, x + nsize[0], y + nsize[1]))
    x = random.randint(0, img.size[0] - size[0])
    y = random.randint(0, img.size[1] - size[1])
    bg_img = img.crop((x, y, x + size[0], y + size[1]))
    x = random.randint(0, 1) * size[1] // 2
    y = random.randint(0, 1) * size[1] // 2
    nbg_img = bg_img.crop((x, y, x + nsize[0], y + nsize[1]))
    blend_img = Image.blend(nbg_img, fg_img, alpha)
    bg_img.paste(blend_img, (x, y))
    return bg_img


class RandomMultiImageAugDataset(Dataset):

    def __init__(self, dataset, n_classes, aug_fn=basic_random_blend, min_n_images_per_class=100, p=0.5):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.indices_per_label = np.zeros((n_classes, len(self.dataset)), dtype=np.bool)
        self.indices = np.arange(len(self.dataset))
        self.min_n_images_per_class = min_n_images_per_class
        self._aug_fn = aug_fn
        self.p = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, y = self.dataset[index]
        indices = self.indices[self.indices_per_label[y, :]]
        self.indices_per_label[y, index] = True
        if len(indices) < self.min_n_images_per_class:
            return img, y

        if random.random() > self.p:
            return img, y

        rindex = random.choice(list(indices))
        rimg, ry = self.dataset[rindex]
        assert y == ry, "{} vs {} at index {}".format(ry, y, rindex)
        nimg = self.aug(img, rimg)
        return nimg, y

    def aug(self, img, rimg):
        assert isinstance(img, Image.Image) and isinstance(rimg, Image.Image), \
            "Input images should be PIL.Image, but given {} and {}".format(img, rimg)
        return self._aug_fn(img, rimg)
