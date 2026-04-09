import torch
from torch import nn
from torch.autograd import Variable
import copy
#from model.masked_cross_entropy import *

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model):

        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {} 
        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = variable(p.data)

            
    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if n.startswith("v_head."):
                continue
            _loss = (p - self._means[n]) ** 2
            loss += _loss.sum()
            input()

        return loss
