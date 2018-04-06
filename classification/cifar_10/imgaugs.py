
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomChoice, RandomAffine
from torchvision.transforms import ColorJitter


train_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(hue=0.05, brightness=0.05)
]


val_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
]

