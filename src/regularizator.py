import torch


class Regularizator:

    def __init__(self, model, coefficients, device, reg_function):
        self.coefficients = [torch.as_tensor(x).to(device) for x in coefficients]
        self.model = model
        self.device = device
        self.reg_function = reg_function
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

    def regularize(self, loss):
        for params in self.elems:
            weights, bias, norm_coef, norm_bias = params
            loss += self.reg_function(weights,
                                      bias,
                                      norm_coef,
                                      norm_bias,
                                      self.coefficients,
                                      self.device)
