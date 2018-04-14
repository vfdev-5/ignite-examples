from torch.nn import Module, Linear, ModuleList

from pretrainedmodels.models.nasnet import nasnetalarge


class FurnitureNASNetALarge(Module):

    def __init__(self, pretrained):
        super(FurnitureNASNetALarge, self).__init__()

        self.model = nasnetalarge(num_classes=1000, pretrained=pretrained)
        filters = self.model.penultimate_filters // 24
        self.model.last_linear = Linear(24*filters, 128)

        for m in self.model.last_linear.modules():
            if isinstance(m, Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # create aliases:
        self.stem = ModuleList([
            self.model.conv0,
            self.model.cell_stem_0,
            self.model.cell_stem_1,
        ])
        self.features = ModuleList([
            self.model.cell_0,
            self.model.cell_1,
            self.model.cell_2,
            self.model.cell_3,
            self.model.cell_4,
            self.model.cell_5,
            self.model.reduction_cell_0,
            self.model.cell_6,
            self.model.cell_7,
            self.model.cell_8,
            self.model.cell_9,
            self.model.cell_10,
            self.model.cell_11,
            self.model.reduction_cell_1,
            self.model.cell_12,
            self.model.cell_13,
            self.model.cell_14,
            self.model.cell_15,
            self.model.cell_16,
            self.model.cell_17
        ])
        self.classifier = self.model.last_linear

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.logits(x)
        return x
