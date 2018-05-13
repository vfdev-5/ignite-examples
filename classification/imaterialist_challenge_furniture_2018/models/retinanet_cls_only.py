import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50


class FurnitureRetinaNetClassification(nn.Module):
    num_anchors = 1

    def __init__(self, num_classes, pretrained=True):
        super(FurnitureRetinaNetClassification, self).__init__()
        self.fpn = FPN50(pretrained)
        self.num_classes = num_classes
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

        n_levels = 5
        self.inner_classifier = nn.Linear(n_levels * num_classes, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.4)
        self.final_classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        cls_preds = []
        for fm in fms:
            cls_pred = self.cls_head(fm)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            cls_pred = F.relu(cls_pred)
            cls_pred, _ = torch.max(cls_pred, dim=1)
            cls_preds.append(cls_pred)

        cls_preds = torch.cat(cls_preds, dim=1)
        out = self.inner_classifier(cls_preds)
        out = self.drop(out)
        out = self.final_classifier(out)
        return out

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


class FPN50(nn.Module):
    def __init__(self, pretrained):
        super(FPN50, self).__init__()

        self.resnet = resnet50(pretrained=pretrained)

        self.stem = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        self.low_features = nn.Sequential(
            self.resnet.layer1,
            self.resnet.layer2,
        )
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Aliases
        self.mid_features = nn.ModuleList([
            self.layer3,
            self.layer4,
        ])

        self.top_features = nn.ModuleList([
            self.conv6,
            self.conv7,
            self.toplayer,
            self.latlayer1,
            self.latlayer2,
            self.smooth1,
            self.smooth2
        ])

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, h, w = y.size()
        return F.upsample(x, size=(h, w), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = self.stem(x)
        c3 = self.low_features(c1)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7
