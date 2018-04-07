
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, ColorJitter


train_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(hue=0.1, brightness=0.1),
]


val_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
]

test_imgaugs = val_imgaugs
