import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

    # @title Default title text
    # Backbone Network:
    import torch
    import os
    import pandas as pd
    from skimage import io, transform
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, utils, models
    import numpy as np
    import pandas as pd
    import shutil, time, os, requests, random, copy
    from itertools import permutations
    import seaborn as sns

    import math

    import imageio
    from skimage.transform import rotate, AffineTransform, warp, resize
    # import skvideo.io as vidio
    # from google.colab.patches import cv2_imshow
    from IPython.display import clear_output, Image, SVG
    import h5py
    # from tabulate import tabulate

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from sklearn.utils import shuffle
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, accuracy_score
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.model_selection import train_test_split

    import matplotlib.pyplot as plt

    try:
        from torch.hub import load_state_dict_from_url
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url

    from typing import Type, Any, Callable, Union, List, Optional
    from torch import Tensor
    from collections import OrderedDict

    import torch
    from torch import Tensor
    import torch.nn as nn
    from torch.hub import load_state_dict_from_url
    # from torch.utils import load_state_dict_from_url
    from typing import Type, Any, Callable, Union, List, Optional
    import torch.nn.functional as F

    __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
               'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
               'wide_resnet50_2', 'wide_resnet101_2']

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }

    def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
        """3x3 convolution with padding"""
        return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
        """1x1 convolution"""
        return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    class BasicBlock(nn.Module):
        expansion: int = 1

        def __init__(
                self,
                inplanes: int,
                planes: int,
                stride: int = 1,
                downsample: Optional[nn.Module] = None,
                groups: int = 1,
                base_width: int = 64,
                dilation: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = None
        ) -> None:
            super(BasicBlock, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x: Tensor) -> Tensor:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class Bottleneck(nn.Module):
        # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
        # while original implementation places the stride at the first 1x1 convolution(self.conv1)
        # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
        # This variant is also known as ResNet V1.5 and improves accuracy according to
        # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

        expansion: int = 4

        def __init__(
                self,
                inplanes: int,
                planes: int,
                stride: int = 1,
                downsample: Optional[nn.Module] = None,
                groups: int = 1,
                base_width: int = 64,
                dilation: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = None
        ) -> None:
            super(Bottleneck, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x: Tensor) -> Tensor:
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class ResNet2D1(nn.Module):

        def __init__(
                self,
                block: Type[Union[BasicBlock, Bottleneck]],
                layers: List[int],
                num_classes: int = 20,
                zero_init_residual: bool = False,
                groups: int = 1,
                input_size: int = 3,
                width_per_group: int = 64,
                replace_stride_with_dilation: Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., nn.Module]] = None
        ) -> None:
            super(ResNet2D1, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv1d(input_size, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=4,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=8,
                                           dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(512 * block.expansion, 256)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

            self.hash_fc = nn.Sequential(
                nn.Linear(256, 64, bias=False),
                nn.BatchNorm1d(64, momentum=0.1)
            )
            self.fcn = nn.Linear(64, num_classes);
            nn.init.normal_(self.hash_fc[0].weight, std=0.01)
            # nn.init.zeros_(self.hash_fc.bias)

        def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                        stride: int = 1, dilate: bool = False) -> nn.Sequential:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

            return nn.Sequential(*layers)

        def _forward_impl(self, x: Tensor) -> Tensor:
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            # print(x.size())
            # x = x.mean(dim=2)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            x = self.fc(x)
            x = F.relu(x)
            x = self.hash_fc(x)
            o = self.fcn(x)
            o = F.softmax(o, dim=1)
            return o, x

        def forward(self, x: Tensor) -> Tensor:
            return self._forward_impl(x)

    def _resnet(
            arch: str,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            pretrained: bool,
            progress: bool,
            **kwargs: Any
    ) -> ResNet2D1:
        model = ResNet2D1(block, layers, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            model.load_state_dict(state_dict)
        return model

    def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""ResNet-18 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                       **kwargs)

    def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""ResNet-50 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                       **kwargs)

    def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""ResNet-152 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                       **kwargs)

    def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""ResNet-34 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                       **kwargs)

    def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""ResNet-101 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                       **kwargs)

    def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""ResNeXt-50 32x4d model from
        `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 4
        return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                       pretrained, progress, **kwargs)

    def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""ResNeXt-101 32x8d model from
        `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                       pretrained, progress, **kwargs)

    def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""Wide ResNet-50-2 model from
        `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
        The model is the same as ResNet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
        channels, and in Wide ResNet-50-2 has 2048-1024-2048.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs['width_per_group'] = 64 * 2
        return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                       pretrained, progress, **kwargs)

    def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet2D1:
        r"""Wide ResNet-101-2 model from
        `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
        The model is the same as ResNet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
        channels, and in Wide ResNet-50-2 has 2048-1024-2048.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs['width_per_group'] = 64 * 2
        return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                       pretrained, progress, **kwargs)

    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader

    class Data(Dataset):
        def __init__(self, X_train, y_train):
            # need to convert float64 to float32 else
            # will get the following error
            # RuntimeError: expected scalar type Double but found Float
            self.X = torch.from_numpy(X_train.astype(np.float32))
            # need to convert float64 to Long else
            # will get the following error
            # RuntimeError: expected scalar type Long but found Float
            self.y = torch.from_numpy(y_train).type(torch.LongTensor)
            self.len = self.X.shape[0]

        def __getitem__(self, index):
            return self.X[index], self.y[index]

        def __len__(self):
            return self.len

        traindata = Data(train_data, train_label)

        batch_size = 1024

        trainloader = DataLoader(traindata, batch_size=batch_size,
                                 shuffle=True, num_workers=2)

'''
import numpy as np
from sklearn.model_selection import train_test_split

training_data, validation_data, training_label, validation_label = train_test_split(
   train_data, train_label, test_size=0.70, random_state=42, shuffle=True, stratify=None)
'''



import torch.optim as optim

model_resnet = resnet18()
criterion = nn.CrossEntropyLoss()

runSAM = False
optimizer = 'sgd'
#optimizer = 'adam'
lr = 0.01
rho = 0.05
nEpoch = 100




if runSAM==False:
  optimizer = torch.optim.SGD(model_resnet.parameters(), lr=lr, momentum=0.9)
  #optimizer = optim.Adam(model_resnet.parameters(), lr=1e-8)
else:
  base_optimizer = torch.optim.SGD # define an optimizer for the "sharpness-aware" update
  optimizer = SAM(model_resnet.parameters(), base_optimizer, lr=lr, momentum=0.9, rho=rho)


print(optimizer)


# val data Loader
#val_data = Data(validation_data, validation_label)

#batch_size = len(validation_data)
#validationloader = DataLoader(val_data, batch_size=batch_size,
#                         shuffle=False, num_workers=2)



# test data loader

#test_data = Data(validation_data, validation_label)

#batch_size = len(validation_data)
#validationloader = DataLoader(val_data, batch_size=batch_size,
#                         shuffle=False, num_workers=2)


#100 epoch for batch = 1024*
best_loss = 10000 #smaller is better.
max_limit = 20
counter = 0
#model_resnet = resnet18()

model_resnet = model_resnet.cuda()

#optimizer = optim.Adam(model_resnet.parameters(), lr=1e-3)


for epoch in range(nEpoch):  # loop over the dataset multiple times
    running_loss = 0.0
    val_running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # print(inputs.shape)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs1 = model_resnet(torch.tensor(inputs).transpose(1,2).float())
        outputs = outputs1[0]#1
        # print(outputs.shape)

        # print(type(outputs))
        labels = torch.squeeze(labels, dim=1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


    print("Epoch:", epoch+1, "-->","train loss: ",running_loss, " --> Validation loss: ", val_running_loss)

print('Finished Training')

'''
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(inputs.shape)
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        val_outputs = model_resnet(torch.tensor(inputs).transpose(1,2).float())
        val_output = val_outputs[0]

        # labels = torch.squeeze(torch.tensor(validation_label), dim=1)
        labels = torch.squeeze(labels, dim=1)


        val_loss = criterion(val_output, labels)
        # val_loss.backward()
        # optimizer.step()

        # print statistics
        val_running_loss += val_loss.item()
    #if val_running_loss > best_loss:    #0.1, 0

    if best_loss > val_running_loss:
        best_loss = val_running_loss  #0.1


        counter=0
        PATH = './drive/MyDrive/model.pth'
        torch.save(model_resnet.state_dict(), PATH)
    else:
        counter+=1
'''

test_data = torch.from_numpy(test_data).float()
test_data = test_data.cuda()

pred, embed = model_resnet(test_data.transpose(1,2).float())
correct = 0
total = 0
labels = torch.squeeze(torch.from_numpy(test_label), dim=1)
_, predicted = torch.max(pred.data, 1)
total = labels.size(0)
correct = (predicted == labels.cuda()).sum().item()
acc = correct/total
print("Final Accuracy: ",acc)

# sam
#100 epoch for batch = 1024*
best_loss = 10000 #smaller is better.
max_limit = 20
counter = 0
#model_resnet = resnet18()

train_data_error = torch.tensor(training_data_error).cuda().transpose(1,2).float()
error_label = torch.tensor(error_label).cuda()
model_resnet = model_resnet.cuda()
output, embed = model_resnet(train_data_error)
loss_forget_init = criterion(output[:n,:], error_label[:n,:].squeeze())
print(loss_forget_init)
#optimizer = optim.Adam(model_resnet.parameters(), lr=1e-3)

#model_resnet = resnet18().cuda()
#base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
#optimizer = SAM(model_resnet.parameters(), base_optimizer, lr=0.1, momentum=0.9)

criterion = nn.CrossEntropyLoss()

for epoch in range(nEpoch):  # loop over the dataset multiple times
    running_loss = 0.0
    val_running_loss = 0.0
    for i, data in enumerate(trainloader_w_err, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # print(inputs.shape)
       # print(inputs.shape)

        # first forward-backward step
        enable_running_stats(model_resnet)# <- this is the important line


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs1 = model_resnet(torch.tensor(inputs).transpose(1,2))
        outputs = outputs1[0]#1
        # print(outputs.shape)

        # print(type(outputs))
        labels = torch.squeeze(labels, dim=1)
        #def closure():
        #  loss = criterion(outputs, labels)
        #  loss.backward()
        #  return loss

        loss = criterion(outputs, labels)
        loss.backward()
        #optimizer.step(closure)
        optimizer.first_step(zero_grad= True)
        # second forward-backward step
        disable_running_stats(model_resnet)  # <- this is the important line
        tmp = criterion(model_resnet(torch.tensor(inputs).transpose(1,2).float())[0], labels)
        tmp.backward()
        optimizer.second_step(zero_grad=True)


        optimizer.zero_grad()

        # print statistics
        running_loss += loss.item()
        print("Epoch:", epoch+1, "-->", running_loss, loss.item(), tmp.item())
        #print("Epoch:", epoch+1, "-->","train loss: ",loss.item(), "second loss: ", tmp.item())
'''

    for i, data in enumerate(validationloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print(inputs.shape)
        inputs = inputs.float().cuda()
        labels = labels.float().cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        val_outputs = model_resnet(torch.tensor(inputs).transpose(1,2).float())
        val_output = val_outputs[0]

        # labels = torch.squeeze(torch.tensor(validation_label), dim=1)
        labels = torch.squeeze(labels, dim=1)


        val_loss = criterion(val_output, labels)
        # val_loss.backward()
        # optimizer.step()

        # print statistics
        val_running_loss += val_loss.item()
    #if val_running_loss > best_loss:    #0.1, 0

    if best_loss > val_running_loss:
        best_loss = val_running_loss  #0.1


        counter=0
        PATH = './drive/MyDrive/model.pth'
        torch.save(model_resnet.state_dict(), PATH)
    else:
        counter+=1
    output, embed = model_resnet(train_data_error)
    loss_forget_init = criterion(output[:n,:], error_label[:n,:].squeeze())
    loss_normal = criterion(output[n:,:], error_label[n:,:].squeeze())

    print("Epoch:", epoch+1, "-->","train loss: ",running_loss, " --> Validation loss: ", val_running_loss, " --> Best loss: ", best_loss,"--> Noisy: ",loss_forget_init.item(),"--> Actual: ",loss_normal.item())
'''
print('Finished Training')


# prompt: compute accuracy of vaildation data

model_resnet.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data in validationloader:
        inputs, labels = data
        # print(inputs.shape)
        inputs = inputs.cuda()
        labels = labels.cuda()


        # forward + backward + optimize
        val_outputs = model_resnet(torch.tensor(inputs).transpose(1,2).float())
        val_output = val_outputs[0]

        # labels = torch.squeeze(torch.tensor(validation_label), dim=1)
        labels = torch.squeeze(labels, dim=1)
        _, predicted = torch.max(val_output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on the validation: %d %%' % (100 * correct / total))