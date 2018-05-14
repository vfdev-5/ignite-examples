from torch.nn import Module, Linear, ModuleList, AdaptiveAvgPool2d, Sequential
from torch.nn.init import normal_, constant_
from torchvision.models.resnet import *
from .resnet_v2 import *


class FurnitureResNet152_350(Module):

    def __init__(self, pretrained=True):
        super(FurnitureResNet152_350, self).__init__()

        self.model = resnet152(num_classes=1000, pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = Linear(num_features, 128)
        self.model.avgpool = AdaptiveAvgPool2d(1)

        for m in self.model.fc.modules():
            if isinstance(m, Linear):
                normal_(m.weight, 0, 0.01)
                constant_(m.bias, 0)

        # create aliases:
        self.stem = ModuleList([
            self.model.conv1,
            self.model.bn1,
        ])
        self.features = ModuleList([
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ])
        self.classifier = self.model.fc

    def forward(self, x):
        return self.model(x)


class FurnitureResNet152_350_FC(Module):

    def __init__(self, pretrained=True):
        super(FurnitureResNet152_350_FC, self).__init__()

        self.model = resnet152(num_classes=1000, pretrained=pretrained)
        self.model.avgpool = AdaptiveAvgPool2d(1)

        # create aliases:
        self.stem = ModuleList([
            self.model.conv1,
            self.model.bn1,
            self.model.layer1,
        ])
        self.features = ModuleList([
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ])
        self.classifier = self.model.fc

        self.final_classifiers = Sequential(
            Linear(1000, 1000),
            Linear(1000, 128),
        )

    def forward(self, x):
        x = self.model(x)
        return self.final_classifiers(x)


class FurnitureResNet101_350_finetune(Module):

    def __init__(self, pretrained=True):
        super(FurnitureResNet101_350_finetune, self).__init__()

        self.model = resnet101(num_classes=1000, pretrained=pretrained)
        self.model.avgpool = AdaptiveAvgPool2d(1)

        # create aliases:
        self.stem = ModuleList([
            self.model.conv1,
            self.model.bn1,
            self.model.layer1,
            self.model.layer2,
        ])
        self.features = ModuleList([
            self.model.layer3,
            self.model.layer4,
        ])
        self.classifier = self.model.fc

        self.final_classifiers = Sequential(
            Linear(1000, 1000),
            Linear(1000, 128),
        )

        for m in self.final_classifiers.modules():
            if isinstance(m, Linear):
                normal_(m.weight, 0, 0.01)
                constant_(m.bias, 0)

        # freeze internal layers:
        for param in self.stem.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return self.final_classifiers(x)
    
    
class FurnitureResNet101_350_finetune2(Module):

    def __init__(self, pretrained=True):
        super(FurnitureResNet101_350_finetune2, self).__init__()

        self.model = resnet101(num_classes=1000, pretrained=pretrained)
        self.model.avgpool = AdaptiveAvgPool2d(1)

        # create aliases:
        self.stem = ModuleList([
            self.model.layer2,
            self.model.layer3,            
        ])
        self.features = ModuleList([
            self.model.layer4,
        ])
        self.classifier = self.model.fc

        self.final_classifiers = Sequential(
            Linear(1000, 1000),
            Linear(1000, 128),
        )

        for m in self.final_classifiers.modules():
            if isinstance(m, Linear):
                normal_(m.weight, 0, 0.01)
                constant_(m.bias, 0)

        # freeze internal layers:
        for param in self.model.conv1.parameters():
            param.requires_grad = False
        for param in self.model.bn1.parameters():
            param.requires_grad = False
        for param in self.model.layer1.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.model(x)
        return self.final_classifiers(x)    


class FurnitureResNet50_350_finetune(Module):

    def __init__(self, pretrained=True):
        super(FurnitureResNet50_350_finetune, self).__init__()

        self.model = resnet50(num_classes=1000, pretrained=pretrained)
        self.model.avgpool = AdaptiveAvgPool2d(1)

        # create aliases:
        self.stem = ModuleList([
            self.model.conv1,
            self.model.bn1,
            self.model.layer1,
            self.model.layer2,
        ])
        self.features = ModuleList([
            self.model.layer3,
            self.model.layer4,
        ])
        self.classifier = self.model.fc

        self.final_classifiers = Sequential(
            Linear(1000, 1000),
            Linear(1000, 128),
        )

        for m in self.final_classifiers.modules():
            if isinstance(m, Linear):
                normal_(m.weight, 0, 0.01)
                constant_(m.bias, 0)

        # freeze internal layers:
        for param in self.stem.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return self.final_classifiers(x)


class FurnitureResNetV2_50_350(Module):

    def __init__(self):
        super(FurnitureResNetV2_50_350, self).__init__()

        self.model = resnet50_v2(num_classes=1000)
        num_features = self.model.fc.in_features
        self.model.fc = Linear(num_features, 128)

        for m in self.model.fc.modules():
            if isinstance(m, Linear):
                normal_(m.weight, 0, 0.01)
                constant_(m.bias, 0)

        # create aliases:
        self.stem = ModuleList([
            self.model.conv1,
            self.model.bn1,
        ])
        self.features = ModuleList([
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.bn5
        ])
        self.classifier = self.model.fc

    def forward(self, x):
        return self.model(x)


