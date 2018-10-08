import mmd
import torchvision.models as models

from torch import nn

class Alexnet_finetune(nn.Module):
    def __init__(self, num_classes=31):
        super(Alexnet_finetune, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.final_classifier(x)
        return output


class DCCNet(nn.Module):
    def __init__(self, num_classes=31):
        super(DCCNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True)
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def forward(self, source, target):
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        source = self.bottleneck(source)

        mmd_loss = 0
        if self.training:
            target = self.features(target)
            target = target.view(target.size(0), -1)
            target = self.classifier(target)
            target = self.bottleneck(target)
            mmd_loss += mmd.mmd_linear(source, target)

        result = self.final_classifier(source)

        return result, mmd_loss

def load_pretrained_alexnet(model):
    """
    load pretrained alexnet parameters into defined ddcnet
    :param ddcnet: defined ddcnet model
    :return: defined ddcnet model with pretrained alexnet parameters
    """
    alexnet = models.alexnet(pretrained=True)
    pretrained_dict = alexnet.state_dict()
    model_dict = model.state_dict()

    for key, value in model_dict.items():
        if key.split('.')[0] in ['features', 'classifier']:
            model_dict[key] = pretrained_dict[key]
    model.load_state_dict(model_dict)

    return model
