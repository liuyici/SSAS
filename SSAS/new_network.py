
from typing import Optional, Tuple
from torch.nn import Parameter
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from sklearn.metrics import  precision_score, recall_score, f1_score, roc_curve, auc,cohen_kappa_score, accuracy_score
import torch.nn.functional as F


import sys



def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r * torch.tanh(self.bias))
        output = cosine
        return output



class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).cuda()

    def forward(self, x):
        if self.training:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x



class MLPBase(nn.Module):
    """
    The MLP based feature extraction module

    Parameters:
        - **input_size** (int): num of input features
        - **hidden_sizehidden_size** Optional(int): num of output features
    """
 
    def __init__(self,
                 input_size: int,
                 hidden_size: Optional[int] = None):
        super(MLPBase, self).__init__()
        # layer1
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size, momentum=0.1, affine=False),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            GaussianNoise(1.0)
        )
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            GaussianNoise(1.0)
        )
    
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
             
      x = x.view(x.size(0), -1)
      x = self.feature_extractor(x)
      return x


class feat_bottleneck(nn.Module):
    """
    The bottleneck module

    Parameters:
        - **input_dim** (int): the dim of input feature
        - **bottleneck_dim** (int): the dim of bottleneck module output
        - **t** Optional(float): the temperature factor
    """

    def __init__(self,
                 hidden_size: int,
                 bottleneck_dim: int,
                 use_bottleneck=True,
                 radius=10.0, 
                 t: Optional[float] = 0.1):
        # self, input_size, hidden_size, use_bottleneck=True, bottleneck_dim=100, radius=10.0, class_num=1000
        super(feat_bottleneck, self).__init__()
            # set
        self.use_bottleneck = use_bottleneck
    
        self.bottleneck_fc = nn.Linear(hidden_size, bottleneck_dim)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.bottleneck_fc.apply(init_weights)
        self.__in_features = bottleneck_dim


        self.radius = radius
        # self.radius = nn.Parameter(torch.tensor(radius, requires_grad=True))


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
             # apply bottleneck
        x = self.bottleneck(x)
        x = self.radius * x / (torch.norm(x, dim=1, keepdim=True) + 1e-10)
        return x


class feat_classifier(nn.Module):
    """
    The classifier module

    Parameters:
        - **input_dim** (int): the dim of input feature
        - **n_class** (int): the num of task classes
    """

    def __init__(self,
                 bottleneck_dim: int,
                 class_num: int):
        super(feat_classifier, self).__init__()
        # self.fc = weight_norm(nn.Linear(input_dim, n_class), name="weight")
        self.fc = SLR_layer(bottleneck_dim, class_num)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

