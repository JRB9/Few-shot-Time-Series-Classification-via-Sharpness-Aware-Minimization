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

    import torch

    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super(SAM, self).__init__(params, defaults)

            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)

        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

            if zero_grad: self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

            self.base_optimizer.step()  # do the actual "sharpness-aware" update

            if zero_grad: self.zero_grad()

        @torch.no_grad()
        def step(self, closure=None):
            assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
            closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

            self.first_step(zero_grad=True)
            closure()
            self.second_step()

        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][
                0].device  # put everything on the same device, in case of model parallelism
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
            return norm

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)
            self.base_optimizer.param_groups = self.param_groups

            # mount drive

            from google.colab import drive
            drive.mount('/content/drive')

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

  # change dataset here:

  data_dir = 'drive/MyDrive/classification_data/CharacterTrajectories/'
  shot_dir = '30-shot/'
  train_data = np.load(data_dir + shot_dir + '/X_train.npy')
  train_label = np.load(data_dir + shot_dir + '/y_train.npy')

  # train_data = np.load(data_dir+shot_dir+'/X_train.npy')

  test_data = np.load(data_dir + 'X_test.npy')
  test_label = np.load(data_dir + 'y_test.npy')

print(train_data.shape) # dimension of train_data

print(train_label.shape) # dimension of train_labels

print(np.unique(train_label)) # getting how many classes

print(len(np.unique(train_label)))

traindata = Data(train_data, train_label)

batch_size = 1024

trainloader = DataLoader(traindata, batch_size=batch_size,
                         shuffle=True, num_workers=2)

import torch.optim as optim

model_resnet = resnet18()
criterion = nn.CrossEntropyLoss()

runSAM = True
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


# sam
#100 epoch for batch = 1024*
best_loss = 10000 #smaller is better.
max_limit = 20
counter = 0
#model_resnet = resnet18()


model_resnet = model_resnet.cuda()

#optimizer = optim.Adam(model_resnet.parameters(), lr=1e-3)

#model_resnet = resnet18().cuda()
#base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
#optimizer = SAM(model_resnet.parameters(), base_optimizer, lr=0.1, momentum=0.9)

criterion = nn.CrossEntropyLoss()

for epoch in range(nEpoch):  # loop over the dataset multiple times
    running_loss = 0.0
    val_running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
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
