import torch

ckpt = torch.load('exp/voc2012/pspnet50_normal_split0/model/train_epoch_50.pth')
del ckpt['state_dict']['module.aux.4.weight']
del ckpt['state_dict']['module.aux.4.bias']
new_weight = torch.rand((2, 512, 1, 1))
torch.nn.init.normal_(new_weight, 0, 0.01)
new_weight[0] = ckpt['state_dict']['module.cls.4.weight'][0]
ckpt['state_dict']['module.cls.4.weight'] = new_weight
new_bias = torch.zeros(2)
new_bias[0] = ckpt['state_dict']['module.cls.4.bias'][0]
ckpt['state_dict']['module.cls.4.bias'] = new_bias
torch.save(ckpt, 'exp/voc2012/pspnet50_normal_split0_fs/model/train_epoch_50.pth')