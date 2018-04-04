from torchvision.models.squeezenet import squeezenet1_1


def get_small_squeezenet_v1_1(num_classes):
    from torch.nn import Conv2d, AdaptiveAvgPool2d, Sequential
    model = squeezenet1_1(num_classes=num_classes, pretrained=False)
    # As input image size is small 64x64, we modify first layers:
    # replace : Conv2d(3, 64, (3, 3), stride=(2, 2)) by Conv2d(3, 64, (3, 3), stride=(1, 1), padding=1)
    # remove : MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1)))
    layers = [l for i, l in enumerate(model.features) if i != 2]
    layers[0] = Conv2d(3, 64, kernel_size=(3, 3), padding=1)
    model.features = Sequential(*layers)
    # Replace the last AvgPool2d -> AdaptiveAvgPool2d
    layers = [l for l in model.classifier]
    layers[-1] = AdaptiveAvgPool2d(1)
    model.classifier = Sequential(*layers)
    return model
