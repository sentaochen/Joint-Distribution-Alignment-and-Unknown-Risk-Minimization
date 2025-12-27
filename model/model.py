import torch
import torch.nn as nn

from model.basenet import network_dict
from loss.loss import RCS_loss
from utils import globalvar as gl

class MODEL(nn.Module):

    def __init__(self, basenet, n_class, bottleneck_dim, source_classes_num):
        super(MODEL, self).__init__()
        self.basenet = network_dict[basenet]()
        self.basenet_type = basenet
        self._in_features = self.basenet.len_feature()
        
        self.bottleneck = nn.Sequential(
            nn.Linear(self._in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        self.fc = nn.Linear(bottleneck_dim, n_class)

        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()
        self.source_classes_num = source_classes_num
        

    def forward(self, source, target=None, source_label=None, train=False):
        DEVICE = gl.get_value('DEVICE')
        source_features = self.basenet(source)
        source_features = self.bottleneck(source_features)
        source_output = self.fc(source_features)
        target_features = self.basenet(target)
        target_features = self.bottleneck(target_features)
        target_output = self.fc(target_features)

        softmax_layer = nn.Softmax(dim=1).to(DEVICE)
        source_softmax = softmax_layer(source_output)
        target_softmax = softmax_layer(target_output)
        # source_prob, source_l = torch.max(target_softmax, 1)
        target_prob, target_l = torch.max(target_softmax, 1)
        if train and torch.sum(target_l<self.source_classes_num) > 0:
            loss = RCS_loss(source_features, source_label, target_features[target_l<self.source_classes_num], target_l[target_l<self.source_classes_num], DEVICE)
        else:
            loss = 0
        return source_output, target_output, source_softmax, target_softmax, loss


        
    
    def get_bottleneck_features(self, inputs):
        features = self.basenet(inputs)
        return self.bottleneck(features)

    def get_fc_features(self, inputs):
        features = self.basenet(inputs)
        features = self.bottleneck(features)
        return self.fc(features)


