import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision.models import VGG16_Weights  # Add this import

class FullVggCompositionNet(nn.Module):
    def __init__(self, pretrained=True, isFreeze=False, 
                 LinearSize1=1024, LinearSize2=512):
        super(FullVggCompositionNet, self).__init__()

        # Replace weights=pretrained with proper weight specification
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, LinearSize1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(LinearSize1, LinearSize2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(LinearSize2, 1),
        )

        if isFreeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_name(self):
        return self.__class__.__name__
