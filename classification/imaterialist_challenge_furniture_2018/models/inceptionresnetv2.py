import torch
import torch.nn as nn
from torch.nn import Module, Linear, ModuleList, AdaptiveAvgPool2d, ReLU, Dropout
from torch.nn.init import normal_, constant_
# from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from pretrainedmodels.models.inceptionresnetv2 import inceptionresnetv2


class FurnitureInceptionResNet299(Module):

    def __init__(self, pretrained=True):
        super(FurnitureInceptionResNet299, self).__init__()

        self.model = inceptionresnetv2(num_classes=1000, pretrained=pretrained)
        self.model.last_linear = Linear(1536, 128)

        for m in self.model.last_linear.modules():
            if isinstance(m, Linear):
                normal_(m.weight, 0, 0.01)
                constant_(m.bias, 0.0)

        # create aliases:
        self.stem = ModuleList([
            self.model.conv2d_1a,
            self.model.conv2d_2a,
            self.model.conv2d_2b,
            self.model.conv2d_3b,
            self.model.conv2d_4a,
        ])
        self.features = ModuleList([
            self.model.mixed_5b,
            self.model.repeat,
            self.model.mixed_6a,
            self.model.repeat_1,
            self.model.mixed_7a,
            self.model.repeat_2,
            self.model.block8,
            self.model.conv2d_7b
        ])
        self.classifier = self.model.last_linear

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.logits(x)
        return x


class FurnitureInceptionResNetOnFiveCrops(Module):

    def __init__(self, pretrained=True, n_cls_layers=1024):
        super(FurnitureInceptionResNetOnFiveCrops, self).__init__()

        self.model = inceptionresnetv2(num_classes=1000, pretrained=pretrained)
        self.model.avgpool_1a = AdaptiveAvgPool2d(1)

        n_crops = 5
        self.crop_classifiers = []
        for i in range(n_crops):
            self.crop_classifiers.append(Linear(1536, n_cls_layers))
            for m in self.crop_classifiers[-1].modules():
                if isinstance(m, Linear):
                    normal_(m.weight, 0, 0.01)
                    constant_(m.bias, 0.0)

        # create aliases:
        self.stem = ModuleList([
            self.model.conv2d_1a,
            self.model.conv2d_2a,
            self.model.conv2d_2b,
        ])
        self.low_features = ModuleList([
            self.model.mixed_5b,
            self.model.repeat,
            self.model.mixed_6a,
            self.model.repeat_1
        ])
        self.features = ModuleList([
            self.model.mixed_7a,
            self.model.repeat_2,
            self.model.block8,
            self.model.conv2d_7b
        ])
        self.crop_classifiers = ModuleList(self.crop_classifiers)

        self.drop = Dropout(p=0.45)
        self.relu = ReLU(inplace=True)

        self.final_classifier = Linear(n_cls_layers, 128)
        for m in self.final_classifier.modules():
            normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                constant_(m.bias, 0.0)

    def logits(self, index, features):
        x = self.model.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.crop_classifiers[index](x)
        return x

    def forward(self, crops):
        batch_size, n_crops, *_ = crops.shape

        # Compute features on the crop 0
        features = self.model.features(crops[:, 0, :, :, :])
        features = self.logits(0, features)
        # Add other features
        for i in range(1, n_crops):
            x = self.model.features(crops[:, i, :, :, :])
            features += self.logits(i, x)
        features = self.relu(features)
        features = self.drop(features)
        return self.final_classifier(features)


# class FurnitureInceptionResNetOnCrops(Module):
#
#     def __init__(self, pretrained=True, n_cls_layers=1024, use_checkpoints=False):
#         super(FurnitureInceptionResNetOnCrops, self).__init__()
#
#         self.model = inceptionresnetv2(num_classes=1000, pretrained=pretrained)
#         self.model.avgpool_1a = AdaptiveAvgPool2d(1)
#
#         n_crops = 6
#         self.crop_classifiers = []
#         for i in range(n_crops):
#             self.crop_classifiers.append(Linear(1536, n_cls_layers))
#             for m in self.crop_classifiers[-1].modules():
#                 if isinstance(m, Linear):
#                     normal_(m.weight, 0, 0.01)
#                     constant_(m.bias, 0.0)
#
#         # create aliases:
#         self.stem = ModuleList([
#             self.model.conv2d_1a,
#             self.model.conv2d_2a,
#             self.model.conv2d_2b,
#         ])
#         self.low_features = ModuleList([
#             self.model.mixed_5b,
#             self.model.repeat,
#             self.model.mixed_6a,
#             self.model.repeat_1
#         ])
#         self.features = ModuleList([
#             self.model.mixed_7a,
#             self.model.repeat_2,
#             self.model.block8,
#             self.model.conv2d_7b
#         ])
#         self.crop_classifiers = ModuleList(self.crop_classifiers)
#
#         self.final_classifier = Linear(n_cls_layers, 128)
#         for m in self.final_classifier.modules():
#             normal_(m.weight, mean=0.0, std=0.01)
#             if m.bias is not None:
#                 constant_(m.bias, 0.0)
#
#         # if use_checkpoints:
#         #     self.forward = self._forward_with_checkpoints
#         #     self.num_chunks = 3
#         # else:
#         #     self.forward = self._forward_no_checkpoints
#
#     def logits(self, index, features):
#         x = self.model.avgpool_1a(features)
#         x = x.view(x.size(0), -1)
#         x = self.crop_classifiers[index](x)
#         return x
#
#     # def _forward_with_checkpoints(self, crops):
#     #
#     #     modules = [module for k, module in self._modules.items()][0]
#     #     input_var = checkpoint_sequential(modules, chunks, input_var)
#     #     input_var = input_var.view(input_var.size(0), -1)
#     #     input_var = self.fc(input_var)
#     #
#     #     batch_size, n_crops, *_ = crops.shape
#     #
#     #
#     #     features = []
#     #     for i in range(n_crops):
#     #         x = self.model.features(crops[:, i, :, :, :])
#     #         features.append(self.logits(i, x))
#     #
#     #
#     #     x = sum(features)
#     #     return self.final_classifier(x)
#
#     def forward(self, crops):
#         batch_size, n_crops, *_ = crops.shape
#
#         # Compute features on the crop 0
#         features = self.model.features(crops[:, 0, :, :, :])
#         features = self.logits(0, features)
#         # Add other features
#         for i in range(1, n_crops):
#             x = self.model.features(crops[:, i, :, :, :])
#             features += self.logits(i, x)
#
#         return self.final_classifier(features)


class FurnitureInceptionResNetV4350SSDLike(nn.Module):
    def __init__(self, num_classes, pretrained='imagenet'):
        super(FurnitureInceptionResNetV4350SSDLike, self).__init__()

        self.extractor = Extractor350(pretrained=pretrained)

        self.num_classes = num_classes
        self.num_anchors = (1, 1, 1, 1)
        self.in_channels = self.extractor.channels

        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.cls_layers += [
                nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes,
                          kernel_size=3, padding=1)
            ]

        n_boxes = sum([i ** 2 for i in self.extractor.featuremap_sizes])
        self.boxes_to_classes = []
        for i in range(num_classes):
            self.boxes_to_classes.append(nn.Linear(n_boxes, 1))

        self.boxes_to_classes = nn.ModuleList(self.boxes_to_classes)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.4)
        self.final_classifier = nn.Linear(num_classes, num_classes)

        for param in self.extractor.low_features.parameters():
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


class Extractor350(nn.Module):
    featuremap_sizes = (20, 9, 1)
    channels = (256, 320, 256)

    def __init__(self, pretrained):
        super(Extractor350, self).__init__()

        model = inceptionresnetv2(pretrained=pretrained)
        self.low_features = nn.Sequential(
            model.conv2d_1a,
            model.conv2d_2a,
            model.conv2d_2b,
            model.maxpool_3a,
            model.conv2d_3b,
            model.conv2d_4a,
            model.maxpool_5a,
            model.mixed_5b,
            model.repeat,
            model.mixed_6a,
            model.repeat_1
        )

        self.mid_features = nn.Sequential(
            model.mixed_7a,
            model.repeat_2,
            model.block8
        )

        self.top_features = nn.Sequential(
            model.conv2d_7b,
            model.avgpool_1a,

        )
        self.smooth1 = nn.Conv2d(1088, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(2080, 320, kernel_size=3, stride=1, padding=1)
        self.top_smooth = nn.Conv2d(1536, 256, kernel_size=1, stride=1, padding=0)

        # Alias
        self.smooth_layers = nn.ModuleList([
            self.smooth1,
            self.smooth2,
            self.top_smooth,
        ])

    def forward(self, x):
        out = []
        x1 = self.low_features(x)
        out.append(self.smooth1(x1))

        x2 = self.mid_features(x1)
        out.append(self.smooth2(x2))

        x3 = self.top_features(x2)
        out.append(self.top_smooth(x3))

        return out


class FurnitureInceptionResNetV4_350_retina(nn.Module):
    def __init__(self, num_classes, pretrained='imagenet'):
        super(FurnitureInceptionResNetV4_350_retina, self).__init__()

        self.extractor = FPN350(pretrained=pretrained)

        self.num_classes = num_classes
        self.in_channels = self.extractor.channels

        self.cls_layers = self._make_head(self.num_classes)

        n_boxes = sum([i ** 2 for i in self.extractor.featuremap_sizes])
        self.boxes_to_classes = []
        for i in range(num_classes):
            self.boxes_to_classes.append(nn.Linear(n_boxes, 1))

        self.boxes_to_classes = nn.ModuleList(self.boxes_to_classes)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.4)
        self.final_classifier = nn.Linear(num_classes, num_classes)

        for param in self.extractor.low_features.parameters():
            param.requires_grad = False

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

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


class FPN350(nn.Module):
    featuremap_sizes = (20, 9, 1)
    channels = (256, 320, 256)

    def __init__(self, pretrained):
        super(FPN350, self).__init__()

        model = inceptionresnetv2(pretrained=pretrained)
        self.low_features = nn.Sequential(
            model.conv2d_1a,
            model.conv2d_2a,
            model.conv2d_2b,
            model.maxpool_3a,
            model.conv2d_3b,
            model.conv2d_4a,
            model.maxpool_5a,
            model.mixed_5b,
            model.repeat,
            model.mixed_6a,
            model.repeat_1
        )

        self.mid_features = nn.Sequential(
            model.mixed_7a,
            model.repeat_2,
            model.block8
        )

        self.top_features = nn.Sequential(
            model.conv2d_7b,
            model.avgpool_1a,

        )
        self.smooth1 = nn.Conv2d(1088, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(2080, 320, kernel_size=3, stride=1, padding=1)
        self.top_smooth = nn.Conv2d(1536, 256, kernel_size=1, stride=1, padding=0)

        # Alias
        self.smooth_layers = nn.ModuleList([
            self.smooth1,
            self.smooth2,
            self.top_smooth,
        ])

    def forward(self, x):
        out = []
        x1 = self.low_features(x)
        out.append(self.smooth1(x1))

        x2 = self.mid_features(x1)
        out.append(self.smooth2(x2))

        x3 = self.top_features(x2)
        out.append(self.top_smooth(x3))

        return out
