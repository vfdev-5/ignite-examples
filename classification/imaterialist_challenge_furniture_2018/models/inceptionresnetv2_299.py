from torch.nn import Module, Linear, ModuleList
# from torch.nn.init import normal
from pretrainedmodels.models.inceptionresnetv2 import inceptionresnetv2


class FurnitureInceptionResNet299(Module):

    def __init__(self, pretrained=True):
        super(FurnitureInceptionResNet299, self).__init__()

        self.model = inceptionresnetv2(num_classes=1000, pretrained=pretrained)
        self.model.last_linear = Linear(1536, 128)

        for m in self.model.last_linear.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # create aliases:
        self.stem = ModuleList([
            self.model.conv2d_1a,
            self.model.conv2d_2a,
            self.model.conv2d_2b,
        ])
        self.features = ModuleList([
            self.model.conv2d_3b,
            self.model.conv2d_4a,
            self.model.mixed_5b,
            self.model.repeat,
            self.model.mixed_6a,
            self.model.repeat_1,
            self.model.mixed_7a,
            self.model.repeat_2,
            self.model.block8,
            self.model.conv2d_7b
        ])
        self.classifier = self.model.last_linear

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.logits(x)
        return x
