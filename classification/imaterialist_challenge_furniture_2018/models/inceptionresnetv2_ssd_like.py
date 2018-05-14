import torch
import torch.nn as nn
from torch.nn import Module, Linear, ModuleList, AdaptiveAvgPool2d, ReLU, Dropout
from torch.nn.init import normal_, constant_
from pretrainedmodels.models.inceptionresnetv2 import inceptionresnetv2


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
        self.stem = nn.Sequential(
            model.conv2d_1a,
            model.conv2d_2a,
            model.conv2d_2b,
            model.maxpool_3a,
            model.conv2d_3b,
            model.conv2d_4a,
            model.maxpool_5a,
        )

        self.low_features_a = nn.Sequential(
            model.mixed_5b,
            model.repeat,
        )

        self.low_features_b = nn.Sequential(
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
        self.smooth2 = nn.Conv2d(1088, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(2080, 320, kernel_size=3, stride=1, padding=1)
        self.top_smooth = nn.Sequential(
            nn.Conv2d(1536, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        )

        # aliases
        self.smooth_layers = nn.ModuleList([
            self.smooth2,
            self.smooth3,
            self.top_smooth,
        ])

    def forward(self, x):
        out = []
        x = self.stem(x)

        x = self.low_features_a(x)

        x = self.low_features_b(x)
        out.append(self.smooth2(x))

        x = self.mid_features(x)
        out.append(self.smooth3(x))

        x = self.top_features(x)
        out.append(self.top_smooth(x))

        return out


class FurnitureInceptionResNetV4350SSDLike_v2(nn.Module):
    def __init__(self, num_classes, pretrained='imagenet'):
        super(FurnitureInceptionResNetV4350SSDLike_v2, self).__init__()

        self.extractor = Extractor350_v2(pretrained=pretrained)

        self.num_classes = num_classes
        self.num_anchors = (1, 1, 1)
        self.in_channels = self.extractor.channels

        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.cls_layers += [
                nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes,
                          kernel_size=3, padding=1),
                nn.Sigmoid()
            ]

        n_levels = len(self.extractor.featuremap_sizes)
        self.boxes_to_classes = []
        for i in range(num_classes):
            self.boxes_to_classes.append(nn.Linear(n_levels, 1))
        self.boxes_to_classes = nn.ModuleList(self.boxes_to_classes)

        self.inner_classifier = nn.Linear(n_levels * num_classes, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.4)
        self.final_classifier = nn.Linear(2 * num_classes, num_classes)

    def forward(self, x):

        cls_preds = []
        xs = self.extractor(x)
        # Transform output feature maps to bbox predictions
        for i, x in enumerate(xs):
            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.view(cls_pred.size(0), -1, self.num_classes)
            # Sum all predictions of all boxes at single level
            cls_pred = torch.sum(cls_pred, dim=1).unsqueeze(1)
            cls_preds.append(cls_pred)

        # Two ways to aggregate
        # A) Predictions from each bbox level are transformed with FC to a single probability
        # for each target class
        cls_preds_a = torch.cat(cls_preds, dim=1)
        merged_cls_preds = []
        for i, m in enumerate(self.boxes_to_classes):
            merged_cls_preds.append(m(cls_preds_a[:, :, i]))
        merged_cls_preds = torch.cat(merged_cls_preds, 1)
        out_a = self.relu(merged_cls_preds)

        # B) Predictions from each bbox level are transformed with FC to a vector of output probabilities
        cls_preds_b = torch.cat(cls_preds, dim=2).squeeze(1)
        out_b = self.inner_classifier(cls_preds_b)
        out_b = self.relu(out_b)

        # Aggregate results:
        out = torch.cat([out_a, out_b], dim=1)

        out = self.drop(out)
        out = self.final_classifier(out)
        return out


class Extractor350_v2(nn.Module):
    featuremap_sizes = (20, 9, 1)
    channels = (256, 256, 256)

    def __init__(self, pretrained):
        super(Extractor350_v2, self).__init__()

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

        self.low_features_a = nn.Sequential(
            model.mixed_5b,
            model.repeat,
        )

        self.low_features_b = nn.Sequential(
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
        self.smooth2 = nn.Sequential(
            nn.Conv2d(1088, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(2080, 320, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.top_smooth = nn.Sequential(
            nn.Conv2d(1536, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # aliases
        self.smooth_layers = nn.ModuleList([
            self.smooth2,
            self.smooth3,
            self.top_smooth,
        ])

    def forward(self, x):
        out = []
        x = self.stem(x)

        x = self.low_features_a(x)

        x = self.low_features_b(x)
        out.append(self.smooth2(x))

        x = self.mid_features(x)
        out.append(self.smooth3(x))

        x = self.top_features(x)
        out.append(self.top_smooth(x))

        return out


class FurnitureInceptionResNetV4350SSDLike_v3(nn.Module):
    def __init__(self, num_classes, pretrained='imagenet'):
        super(FurnitureInceptionResNetV4350SSDLike_v3, self).__init__()

        self.extractor = Extractor350_v3(pretrained=pretrained)

        self.num_classes = num_classes
        self.num_anchors = (1, 1, 1)
        self.in_channels = self.extractor.channels

        self.cls_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.cls_layers += [
                nn.Conv2d(self.in_channels[i], self.num_anchors[i] * self.num_classes,
                          kernel_size=3, padding=1),
            ]

        n_levels = len(self.extractor.featuremap_sizes)
        self.boxes_to_classes = []
        for i in range(num_classes):
            self.boxes_to_classes.append(nn.Linear(n_levels, 1))

        self.boxes_to_classes = nn.ModuleList(self.boxes_to_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.4)
        self.final_classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):

        cls_preds = []
        xs = self.extractor(x)
        # Transform output feature maps to bbox predictions
        for i, x in enumerate(xs):
            cls_pred = self.cls_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.view(cls_pred.size(0), -1, self.num_classes)
            # Sum all predictions of all boxes at single level
            cls_pred = torch.sum(cls_pred, dim=1).unsqueeze(1)
            cls_preds.append(cls_pred)
        cls_preds = torch.cat(cls_preds, dim=1)

        merged_cls_preds = []
        for i, m in enumerate(self.boxes_to_classes):
            merged_cls_preds.append(m(cls_preds[:, :, i]))
        merged_cls_preds = torch.cat(merged_cls_preds, 1)

        out = self.relu(merged_cls_preds)
        out = self.drop(out)
        out = self.final_classifier(out)
        return out


class Extractor350_v3(nn.Module):
    featuremap_sizes = (20, 9, 1)
    channels = (256, 320, 256)

    def __init__(self, pretrained):
        super(Extractor350_v3, self).__init__()

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

        self.low_features_a = nn.Sequential(
            model.mixed_5b,
            model.repeat,
        )

        self.low_features_b = nn.Sequential(
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
            nn.ReLU(),
            model.avgpool_1a,

        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(1088, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(2080, 320, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.top_smooth = nn.Sequential(
            nn.Conv2d(1536, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # aliases
        self.smooth_layers = nn.ModuleList([
            self.smooth2,
            self.smooth3,
            self.top_smooth,
        ])

    def forward(self, x):
        out = []
        x = self.stem(x)

        x = self.low_features_a(x)
        x = self.low_features_b(x)
        out.append(self.smooth2(x))

        x = self.mid_features(x)
        out.append(self.smooth3(x))

        x = self.top_features(x)
        out.append(self.top_smooth(x))

        return out
