from torch.nn import Module, Linear, ModuleList, AdaptiveAvgPool2d, Dropout, ReLU, Sequential
from pretrainedmodels.models.inceptionv4 import inceptionv4


class FurnitureInceptionV4_350(Module):

    def __init__(self, num_classes=128, pretrained=True):
        super(FurnitureInceptionV4_350, self).__init__()

        self.model = inceptionv4(num_classes=1000, pretrained=pretrained)
        self.model.avg_pool = AdaptiveAvgPool2d(1)
        self.model.last_linear = Linear(1536, num_classes)

        for m in self.model.last_linear.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # create aliases:
        self.stem = ModuleList([
            self.model.features[0],
            self.model.features[1],
            self.model.features[2],
        ])
        self.features = ModuleList([self.model.features[i] for i in range(3, len(self.model.features))])
        self.classifier = self.model.last_linear

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.logits(x)
        return x


class FurnitureInceptionV4_350_FC(Module):

    def __init__(self, pretrained=True):
        super(FurnitureInceptionV4_350_FC, self).__init__()

        self.model = inceptionv4(num_classes=1000, pretrained=pretrained)
        self.model.avg_pool = AdaptiveAvgPool2d(1)
        self.final_classifier = Linear(1000, 128)
        self.dropout = Dropout(p=0.5)
        self.relu = ReLU(inplace=True)

        for m in self.final_classifier.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # create aliases:
        self.stem = ModuleList([
            self.model.features[0],
            self.model.features[1],
            self.model.features[2],
        ])
        self.features = ModuleList([self.model.features[i] for i in range(3, len(self.model.features))])
        self.classifier = self.model.last_linear

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.logits(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final_classifier(x)
        return x


class FurnitureInceptionV4_350_FC2(Module):

    def __init__(self, pretrained=True):
        super(FurnitureInceptionV4_350_FC2, self).__init__()

        self.model = inceptionv4(num_classes=1000, pretrained=pretrained)
        self.model.avg_pool = AdaptiveAvgPool2d(1)
        self.final_classifier = Sequential(
            Linear(1000, 512),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(512, 128),
        )

        for m in self.final_classifier.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # create aliases:
        self.stem = ModuleList([
            self.model.features[0],
            self.model.features[1],
            self.model.features[2],
        ])
        self.features = ModuleList([self.model.features[i] for i in range(3, len(self.model.features))])
        self.classifier = self.model.last_linear

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.logits(x)
        x = self.final_classifier(x)
        return x
