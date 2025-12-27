import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision
import torch.nn.functional as F
    
class ResNetF(nn.Module):
    
    def __init__(self):
        super(ResNetF, self).__init__()    
        model_resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self._in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def len_feature(self):
        return self._in_features
    
class DenseNet(nn.Module):
    
    def __init__(self):
        super(DenseNet, self).__init__()    
        model_densenet121 = torchvision.models.densenet121(pretrained=True)
        self.features = model_densenet121.features
        self._in_features = model_densenet121.classifier.in_features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
    
    def len_feature(self):
        return self._in_features

network_dict = {"densenet121": DenseNet,
                "resnet": ResNetF}



class Classifier(nn.Module):

    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.fc(x)
        return out

class ReverseLayerF(Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):

    def __init__(self, input_dims, hidden_dims=3072, output_dims=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, output_dims),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class BaseNet(nn.Module):
    
    def __init__(self, basenet, n_class):
        super(BaseNet, self).__init__()
        self.basenet = network_dict[basenet]()
        self._in_features = self.basenet.len_feature()
        self.fc = nn.Linear(self._in_features, n_class)

        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        features = self.basenet(x)
        source_output = self.fc(features)

        return source_output, None

    def get_features(self, x):
        features = self.basenet(x)

        return features
