from torch.nn import Module, Linear, ModuleList, AdaptiveAvgPool2d
from torchvision.models.resnet import resnet152


class FurnitureResNet152_350(Module):

    def __init__(self, pretrained=True):
        super(FurnitureResNet152_350, self).__init__()

        self.model = resnet152(num_classes=1000, pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = Linear(num_features, 128)
        self.model.avgpool = AdaptiveAvgPool2d(1)

        for m in self.model.fc.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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
