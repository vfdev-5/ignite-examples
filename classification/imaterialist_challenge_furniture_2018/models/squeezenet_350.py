from torch.nn import Module, Sequential, Conv2d, AdaptiveAvgPool2d, ReLU, Dropout, ModuleList, Linear
from torchvision.models.squeezenet import squeezenet1_1
from torch.nn.init import normal_, constant_


class FurnitureSqueezeNet350(Module):

    def __init__(self, pretrained=True):
        super(FurnitureSqueezeNet350, self).__init__()
        model = squeezenet1_1(pretrained=pretrained)
        self.features = model.features

        # Final convolution is initialized differently form the rest
        final_conv = Conv2d(512, 128, kernel_size=1)
        self.classifier = Sequential(
            Dropout(p=0.5),
            final_conv,
            ReLU(inplace=True),
            AdaptiveAvgPool2d(1)
        )

        for m in final_conv.modules():
            normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), 128)


class FurnitureSqueezeNetOnCrops(Module):

    def __init__(self, pretrained=True, n_crops=6):
        super(FurnitureSqueezeNetOnCrops, self).__init__()
        model = squeezenet1_1(pretrained=pretrained)
        self.features = model.features
        self.crop_classifiers = []
        for i in range(n_crops):
            # Final convolution is initialized differently form the rest
            final_conv = Conv2d(512, 512, kernel_size=1, bias=False)
            self.crop_classifiers.append(Sequential(
                Dropout(p=0.5),
                final_conv,
                ReLU(inplace=True),
                AdaptiveAvgPool2d(1)
            ))
            for m in final_conv.modules():
                normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    constant_(m.bias, 0.0)

        self.crop_classifiers = ModuleList(self.crop_classifiers)
        self.final_classifier = Linear(512, 128)

        for m in self.final_classifier.modules():
            normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                constant_(m.bias, 0.0)

    def forward(self, crops):
        batch_size, n_crops, *_ = crops.shape
        features = []
        for i in range(n_crops):
            x = self.features(crops[:, i, :, :, :])
            x = self.crop_classifiers[i](x)
            features.append(x.view(batch_size, -1))
        x = sum(features)
        return self.final_classifier(x)
