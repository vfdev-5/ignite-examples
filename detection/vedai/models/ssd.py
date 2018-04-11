
from customized_torchcv.models.ssd import SSD300


class GeoSSD300(SSD300):

    # Steps define centers of default boxes:
    # w, h feature map size:
    # cx = (w + 0.5) * self.steps[i]
    # cy = (h + 0.5) * self.steps[i]

    steps = (8, 16, 32, 64, 100, 300)
    box_sizes = (3, 6, 11, 16, 21, 26, 31)  # default bounding box sizes for each feature map.
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    fm_sizes = (38, 19, 10, 5, 3, 1)
