
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomChoice, RandomAffine
from torchvision.transforms import ColorJitter


train_imgaugs = [
    RandomChoice([
        RandomAffine(degrees=(-50, 50), translate=(0.05, 0.05)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
    ]),
    ColorJitter(hue=0.1)
]


val_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
]

test_imgaugs = val_imgaugs
