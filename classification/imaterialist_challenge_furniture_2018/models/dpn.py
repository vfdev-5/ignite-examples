from torch.nn import Module, Conv2d, ModuleList
from pretrainedmodels.models.dpn import dpn131


class FurnitureDPN131_350(Module):

    def __init__(self, pretrained=True):
        super(FurnitureDPN131_350, self).__init__()

        self.model = dpn131(num_classes=1000, pretrained=pretrained)
        in_chs = self.model.classifier.in_channels
        self.model.classifier = Conv2d(in_chs, 128, kernel_size=1, bias=True)

        for m in self.model.last_linear.modules():
            if isinstance(m, Conv2d):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # create aliases:
        self.stem = ModuleList([
            self.model.features[0],  # conv1_1, InputBlock
            self.model.features[1],  # conv2_1, DualPathBlock
            self.model.features[2],  # conv2_2, DualPathBlock
            self.model.features[3],  # conv2_3, DualPathBlock
            self.model.features[4],  # conv2_4, DualPathBlock
        ])
        self.features = ModuleList([self.model.features[i] for i in range(5, len(self.model.features))])
        self.classifier = self.model.last_linear

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.logits(x)
        return x
