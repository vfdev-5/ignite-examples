from torch.nn import Module, Sequential, Conv2d, AdaptiveAvgPool2d, ReLU, Dropout, ModuleList
from torchvision.models.squeezenet import squeezenet1_1
from torch.nn.init import normal


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
            normal(m.weight.data, mean=0.0, std=0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), 128)
