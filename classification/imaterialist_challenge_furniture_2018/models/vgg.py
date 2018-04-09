from torch.nn import Module, Sequential, Linear, ReLU, Dropout
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



