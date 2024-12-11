import torch

from regularizations import block_regularization_loss_from_weights


class Prunner:

    def __init__(self, model, coefficients, device):
        self.coefficients = [torch.as_tensor(x).to(device) for x in coefficients]
        self.model = model
        self.device = device
        l = 0
        self.elems = []
        last_elem = []
        for param in self.model.parameters():
            if l % 4 == 0:
                last_elem = []
            last_elem.append(param)
            if l % 4 == 3:
                self.elems.append(last_elem)
            l += 1

    def prune(self, loss):
        for params in self.elems:
            weights, bias, norm_coef, norm_bias = params
            loss += block_regularization_loss_from_weights(weights,
                                                           bias,
                                                           norm_coef,
                                                           norm_bias,
                                                           self.coefficients,
                                                           self.device)
