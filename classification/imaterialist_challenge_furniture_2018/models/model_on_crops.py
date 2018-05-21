from torch.nn import Module, Linear, ModuleList, AdaptiveAvgPool2d, ReLU, Dropout, Sequential
from torch.nn.init import normal_, constant_


class FurnitureModelOnCrops(Module):

    def __init__(self, features, featuremap_output_size, n_cls_layers=512):
        super(FurnitureModelOnCrops, self).__init__()

        self.base_features = features
        self.avgpool = AdaptiveAvgPool2d(1)

        n_crops = 6
        self.crop_classifiers = []
        for i in range(n_crops):
            self.crop_classifiers.append(
                Sequential(
                    ReLU(),
                    Linear(featuremap_output_size, n_cls_layers),
                    ReLU(),
                    Dropout(p=0.4)
                )
            )
            for m in self.crop_classifiers[-1].modules():
                if isinstance(m, Linear):
                    normal_(m.weight, 0, 0.01)
                    constant_(m.bias, 0.0)

        self.crop_classifiers = ModuleList(self.crop_classifiers)

        self.final_classifier = Linear(n_cls_layers, 128)
        for m in self.final_classifier.modules():
            normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                constant_(m.bias, 0.0)

    def logits(self, index, features):
            x = self.avgpool(features)
            x = x.view(x.size(0), -1)
            x = self.crop_classifiers[index](x)
            return x

    def forward(self, crops):
        batch_size, n_crops, *_ = crops.shape

        # Compute features on the crop 0
        features = self.base_features(crops[:, 0, :, :, :])
        features = self.logits(0, features)
        # Add other features
        for i in range(1, n_crops):
            x = self.base_features(crops[:, i, :, :, :])
            features += self.logits(i, x)

        return self.final_classifier(features)
