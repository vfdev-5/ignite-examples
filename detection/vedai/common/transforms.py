import random

import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from customized_torchcv.transforms import resize, random_flip, random_paste, random_crop, random_distort


class TorchcvDetectionRandomTransform:

    def __init__(self, img_size, mode='train', max_ratio=4, fill=(123, 116, 103)):
        assert mode in ('train', 'test')
        self.img_size = img_size
        self.transform = self.transform_train if mode == 'train' else self.transform_test
        self.max_ratio = max_ratio
        self.fill = fill

    def __call__(self, x, y):
        img = x
        boxes = torch.Tensor(y[0])
        labels = torch.LongTensor(y[1])
        return self.transform(img, boxes, labels, self.img_size, max_ratio=self.max_ratio, fill=self.fill)

    @staticmethod
    def transform_train(img, boxes, labels, img_size, max_ratio, fill):
        img = random_distort(img)
        # if random.random() < 0.3:
        #     img, boxes = random_paste(img, boxes, max_ratio=max_ratio, fill=fill)
        img, boxes, labels = random_crop(img, boxes, labels, min_scale=0.8, max_aspect_ratio=1.5)
        img, boxes = resize(img, boxes, size=(img_size, img_size), random_interpolation=False)
        img, boxes = random_flip(img, boxes)
        img = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)
        return img, (boxes, labels)

    @staticmethod
    def transform_test(img, boxes, labels, img_size, **kwargs):
        img, boxes = resize(img, boxes, size=(img_size, img_size))
        img = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])(img)
        return img, (boxes, labels)
