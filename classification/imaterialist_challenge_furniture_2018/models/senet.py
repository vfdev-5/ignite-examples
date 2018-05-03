from torch.nn import Module, Linear, ModuleList, AdaptiveAvgPool2d
from pretrainedmodels.models.senet import senet154


class FurnitureSENet154_350(Module):

    def __init__(self, pretrained=True):
        super(FurnitureSENet154_350, self).__init__()

        self.model = senet154(num_classes=1000, pretrained=pretrained)
        self.model.avg_pool = AdaptiveAvgPool2d(1)
        in_features = self.model.last_linear.in_features
        self.model.last_linear = Linear(in_features, 128)

        for m in self.last_linear.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # create aliases:
        self.stem = ModuleList([
            self.model.layer0,
        ])
        self.features = ModuleList([
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ])
        self.classifier = self.model.last_linear

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.logits(x)
        return x
