from torch.nn import Module, Sequential, Linear, ReLU, Dropout, ModuleList
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
