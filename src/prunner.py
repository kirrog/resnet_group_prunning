import torch


class Prunner:

    def __init__(self, model, coefficients, device, prunner_function):
        self.coefficients = [torch.as_tensor(x).to(device) for x in coefficients]
        self.model = model
        self.device = device
        self.prunner_function = prunner_function
        self.elems = []
        l = 0
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
            loss += self.prunner_function(weights,
                                          bias,
                                          norm_coef,
                                          norm_bias,
                                          self.coefficients,
                                          self.device)
