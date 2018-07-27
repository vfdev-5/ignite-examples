import torch
from torch.nn import Module, Sequential, Linear, ReLU, Dropout, ModuleList
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16_bn


class FurnitureVGG16BN(Module):

    def __init__(self, pretrained=True):
        super(FurnitureVGG16BN, self).__init__()
        model = vgg16_bn(pretrained=pretrained)
        self.features = model.features
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 128),
        )

        for m in self.classifier.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FurnitureVGG16BN224(Module):

    def __init__(self, pretrained=True):
        super(FurnitureVGG16BN224, self).__init__()
        model = vgg16_bn(pretrained=pretrained)
        self._features = model.features
        self.classifier = Sequential(
            Linear(512 * 7 * 7, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(True),
            Dropout(),
            Linear(4096, 128),
        )

        # create aliases:
        # Stem = Block 1
        self.stem = ModuleList([
            self._features[0],
            self._features[1],
            self._features[3],
            self._features[4],
        ])
        # Features = Blocks 2 - 5
        self.features = ModuleList([
            self._features[7],
            self._features[8],
            self._features[10],
            self._features[11],

            self._features[14],
            self._features[15],
            self._features[17],
            self._features[18],
            self._features[20],
            self._features[21],

            self._features[24],
            self._features[25],
            self._features[27],
            self._features[28],
            self._features[30],
            self._features[31],

            self._features[34],
            self._features[35],
            self._features[37],
            self._features[38],
            self._features[40],
            self._features[41],
        ])

        for m in self.classifier.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FurnitureVGG16BN224Finetunned(Module):

    def __init__(self, pretrained=True):
        super(FurnitureVGG16BN224Finetunned, self).__init__()
        model = vgg16_bn(pretrained=pretrained)
        self._features = model.features
        self.classifier = model.classifier
        self.final_classifier = Sequential(
            ReLU(True),
            Dropout(),
            Linear(1000, 128)
        )

        # create aliases:
        # Stem = Block 1
        self.stem = ModuleList([
            self._features[0],
            self._features[1],
            self._features[3],
            self._features[4],
        ])
        # Features = Blocks 2 - 5
        self.features = ModuleList([
            self._features[7],
            self._features[8],
            self._features[10],
            self._features[11],

            self._features[14],
            self._features[15],
            self._features[17],
            self._features[18],
            self._features[20],
            self._features[21],

            self._features[24],
            self._features[25],
            self._features[27],
            self._features[28],
            self._features[30],
            self._features[31],

            self._features[34],
            self._features[35],
            self._features[37],
            self._features[38],
            self._features[40],
            self._features[41],
        ])

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.final_classifier(x)
        return x


class FurnitureVGG16BN256(Module):

    def __init__(self, pretrained=True):
        super(FurnitureVGG16BN256, self).__init__()
        model = vgg16_bn(pretrained=pretrained)
        self._features = model.features
        self.classifier = Sequential(
            Linear(512 * 8 * 8, 1024),
            ReLU(True),
            Dropout(),
            Linear(1024, 1024),
            ReLU(True),
            Dropout(),
            Linear(1024, 128),
        )

        # create aliases:
        # Stem = Block 1
        self.stem = ModuleList([
            self._features[0],
            self._features[1],
            self._features[3],
            self._features[4],
        ])
        # Features = Blocks 2 - 5
        self.features = ModuleList([
            self._features[7],
            self._features[8],
            self._features[10],
            self._features[11],

            self._features[14],
            self._features[15],
            self._features[17],
            self._features[18],
            self._features[20],
            self._features[21],

            self._features[24],
            self._features[25],
            self._features[27],
            self._features[28],
            self._features[30],
            self._features[31],

            self._features[34],
            self._features[35],
            self._features[37],
            self._features[38],
            self._features[40],
            self._features[41],
        ])

        for m in self.classifier.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FurnitureVGG16BNSSD300Like(nn.Module):
    fm_sizes = (9, 5, 3, 1)

    def __init__(self, num_classes, pretrained=True):
        super(FurnitureVGG16BNSSD300Like, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = (1, 1, 1, 1)
        self.in_channels = (512, 256, 256, 256)

        self.extractor = VGG16Extractor300(pretrained=pretrained)
        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.cls_layers += [
                nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes,
                          kernel_size=3, padding=1)
            ]

        n_boxes = sum([i ** 2 for i in self.fm_sizes])
        self.boxes_to_classes = []
        for i in range(num_classes):
            self.boxes_to_classes.append(nn.Linear(n_boxes, 1))

        self.boxes_to_classes = nn.ModuleList(self.boxes_to_classes)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.4)
        self.final_classifier = nn.Linear(num_classes, num_classes)

        for param in self.extractor.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        cls_preds = []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))
        cls_preds = torch.cat(cls_preds, 1)

        merged_cls_preds = []
        for i, m in enumerate(self.boxes_to_classes):
            merged_cls_preds.append(m(cls_preds[:, :, i]))
        merged_cls_preds = torch.cat(merged_cls_preds, 1)

        out = self.relu(merged_cls_preds)
        out = self.drop(out)
        out = self.final_classifier(out)
        return out


class VGG16Extractor300(nn.Module):
    def __init__(self, pretrained):
        super(VGG16Extractor300, self).__init__()

        self.features = vgg16_bn(pretrained=pretrained).features[0:-1]  # Ignore the last max poolling
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.top_features = nn.ModuleList([
            self.conv6,
            self.conv7,
            self.conv8_1,
            self.conv8_2,
            self.conv9_1,
            self.conv9_2,
            self.conv10_1,
            self.conv10_2,
            self.conv11_1,
            self.conv11_2,
        ])

    def forward(self, x):
        hs = []
        h = self.features(x)
        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)  # conv8_2

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)  # conv11_2
        return hs
