import math
import itertools
from functools import partial


import torch
from torch.nn import Module, ModuleList, Conv2d

from torchvision.models.vgg import VGG, make_layers

from customized_torchcv.utils.box import box_iou, box_nms, change_box_order


class AnotherSSD300(Module):

    def __init__(self, n_classes):
        super(AnotherSSD300, self).__init__()
        self.n_classes = n_classes

        # Original: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        # small_vgg_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
        small_vgg_config = [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M']
        model = VGG(make_layers(small_vgg_config, batch_norm=True))
        self.vgg_features = model.features

        # Indices corresponds to the ReLU of block 1 - block 5
        #         indices = [5, 12, 22, 32, 42]
        # Indices corresponds to the ReLU of block 2 - block 5
        # indices = [12, 22, 32, 42]
        indices = [22, 32, 42]

        # feature_map_sizes is the size of feature maps of loc_layers before concat
        # Globally, this defines the number of boxes
        # self.feature_map_sizes = [75, 38, 19, 9]
        self.feature_map_sizes = [38, 19, 9]

        # setup forward hooks:
        self.feature_maps = [None] * len(indices)

        self.handles = []
        for i, index in enumerate(indices):
            hook = partial(self.get_feature_map_hook, index=i)
            self.handles.append(self.vgg_features[index].register_forward_hook(hook))

        # Number of feature maps at each output from the extractor
        #         in_channels = (64, 128, 256, 512, 512)
        # in_channels = (128, 256, 512, 512)
        # in_channels = (32, 64, 128, 256)
        in_channels = (64, 128, 256)
        # Number of anchors is defined by how default boxes are generated:
        # for a single feature map, there are 1 small box, 1 large box and 2 rectangles generated
        #         num_anchors = (4, 4, 4, 4, 4)
        # num_anchors = (4, 4, 4, 4)
        num_anchors = (4, 4, 4)

        loc_layers = []
        cls_layers = []
        for i in range(len(in_channels)):
            loc_layers += [Conv2d(in_channels[i], num_anchors[i] * 4,
                                  kernel_size=3, padding=1, stride=2)]
            cls_layers += [Conv2d(in_channels[i], num_anchors[i] * self.n_classes,
                                  kernel_size=3, padding=1, stride=2)]
        self.loc_layers = ModuleList(loc_layers)
        self.cls_layers = ModuleList(cls_layers)

    def get_feature_map_hook(self, module, input, result, index):
        self.feature_maps[index] = result

    def extractor(self, x):
        _ = self.vgg_features(x)
        return self.feature_maps

    def forward(self, x):
        loc_preds = []
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1,
                                           self.n_classes))

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        return loc_preds, cls_preds


def get_default_boxes(feature_map_sizes, steps, box_sizes, aspect_ratios):
    boxes = []
    for i, fm_size in enumerate(feature_map_sizes):
        for h, w in itertools.product(range(fm_size), repeat=2):
            cx = (w + 0.5) * steps[i]
            cy = (h + 0.5) * steps[i]

            s = box_sizes[i]
            boxes.append((cx, cy, s, s))

            s = math.sqrt(box_sizes[i] * box_sizes[i+1])
            boxes.append((cx, cy, s, s))

            s = box_sizes[i]
            for ar in aspect_ratios[i]:
                boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))
    return torch.Tensor(boxes)  # xywh


class BoxCoder300:

    def __init__(self, feature_map_sizes, steps=None, box_sizes=None):

        dim = 300

        if steps is None:
            steps = [int(dim / s) for s in feature_map_sizes]

        if box_sizes is None:
            box_sizes = [2 + 2**i for i in range(len(feature_map_sizes) + 1)]

        aspect_ratios = ((2,), (2,), (2,), (2,))

        self.default_boxes = get_default_boxes(feature_map_sizes, steps, box_sizes, aspect_ratios)

    def encode(self, boxes, labels):
        """Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0]*anchor_w)
          ty = (y - anchor_y) / (variance[0]*anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
        """
        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1][0]
            return i[j], j

        default_boxes = self.default_boxes  # xywh
        default_boxes = change_box_order(default_boxes, 'xywh2xyxy')

        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]
        index = torch.LongTensor(len(default_boxes)).fill_(-1)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i, j] < 1e-6:
                break
            index[i] = j
            masked_ious[i, :] = 0
            masked_ious[:, j] = 0

        mask = (index < 0) & (ious.max(1)[0] >= 0.5)
        if mask.any():
            index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = change_box_order(default_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:, :2]-default_boxes[:, :2]) / default_boxes[:, 2:] / variances[0]
        loc_wh = torch.log(boxes[:, 2:]/default_boxes[:, 2:]) / variances[1]
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index < 0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45):
        """Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        """
        variances = (0.1, 0.2)
        xy = loc_preds[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = torch.exp(loc_preds[:,2:]*variances[1]) * self.default_boxes[:,2:]
        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes-1):
            score = cls_preds[:, i+1]  # class i corresponds to (i+1) column
            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            keep = box_nms(box, score, nms_thresh)
            boxes.append(box[keep])
            labels.append(torch.LongTensor(len(box[keep])).fill_(i))
            scores.append(score[keep])

        if len(boxes) > 0:
            boxes = torch.cat(boxes, 0)
            labels = torch.cat(labels, 0)
            scores = torch.cat(scores, 0)
        return boxes, labels, scores