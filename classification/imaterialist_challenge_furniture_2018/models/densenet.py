from torch.nn import Module, Linear, ModuleList
from torch.nn import functional as F
from torchvision.models.densenet import densenet161


class FurnitureDenseNet161_350(Module):

    def __init__(self, pretrained=True):
        super(FurnitureDenseNet161_350, self).__init__()

        self.model = densenet161(num_classes=1000, pretrained=pretrained)
        num_features = self.model.classifier.in_features
        self.model.classifier = Linear(num_features, 128)

        for m in self.model.classifier.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # create aliases:
        self.stem = ModuleList([
            self.model.features[0],
            self.model.features[1],
        ])
        self.features = ModuleList([self.model.features[i] for i in range(2, len(self.model.features))])
        self.classifier = self.model.classifier

    def forward(self, x):
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, output_size=1).view(features.size(0), -1)
        out = self.model.classifier(out)
        return out
