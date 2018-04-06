
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from torchvision.transforms.functional import _is_pil_image

train_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
]


val_imgaugs = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
]

