import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrainedmodels.models.inceptionresnetv2 import inceptionresnetv2


class FurnitureInceptionResNet350RetinaLike(nn.Module):
    def __init__(self, num_classes, pretrained='imagenet'):
        super(FurnitureInceptionResNet350RetinaLike, self).__init__()

        self.fpn = InceptionResnetFPN350(pretrained=pretrained)

        self.num_classes = num_classes
        self.cls_head = self._make_head(self.num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.base_classifier = nn.Linear(1536, num_classes)

        self.boxes_classifier = nn.Linear(num_classes, num_classes)
        self.relu = nn.ReLU()
        self.final_classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        c5, fms = self.fpn(x)

        # A) Standard classification:
        out_a = self.avgpool(c5)
        out_a = out_a.view(out_a.size(0), -1)
        out_a = self.base_classifier(out_a)

        # B) Boxes classification:
        cls_preds = 0
        for fm in fms:
            cls_pred = self.cls_head(fm)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
            cls_pred = F.relu(cls_pred)
            cls_pred, _ = torch.max(cls_pred, dim=1)
            cls_preds += cls_pred
        out_b = self.boxes_classifier(cls_preds)

        # Merge A + B
        out = out_a + out_b
        out = self.final_classifier(out)
        return out

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


class InceptionResnetFPN350(nn.Module):

    def __init__(self, pretrained):
        super(InceptionResnetFPN350, self).__init__()

        model = inceptionresnetv2(pretrained=pretrained)
        self.stem = nn.Sequential(
            model.conv2d_1a,
            model.conv2d_2a,
            model.conv2d_2b,
            model.maxpool_3a,
            model.conv2d_3b,
            model.conv2d_4a,
            model.maxpool_5a,
        )

        self.low_features = nn.Sequential(
            model.mixed_5b,
            model.repeat,
        )

        self.layer3 = nn.Sequential(
            model.mixed_6a,
            model.repeat_1
        )

        self.layer4 = nn.Sequential(
            model.mixed_7a,
            model.repeat_2,
            model.block8,
            model.conv2d_7b,
        )

        self.conv6 = nn.Conv2d(1536, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(1536, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1088, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)

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
        return c5, (p3, p4, p5, p6, p7)




