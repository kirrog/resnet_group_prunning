from functools import reduce

import torch


@torch.jit.script
def l1_l2_loss(param, wcl1, wcl2):
    res = torch.sum(torch.abs(param)) * wcl1
    res += torch.sum(param ** 2) * wcl2
    return res


@torch.jit.script
def l1_l2_loss_biased(param, wcl1, wcl2):
    param_biased = param - 1.0
    res = torch.sum(torch.abs(param_biased)) * wcl1
    res += torch.sum(param_biased ** 2) * wcl2
    return res


@torch.jit.script
def entropy_loss(param, coef):
    d = torch.abs(param)
    return - d * torch.log(d) * coef / d.numel()


# iterate by version of oi and calc mean - mat ojid
# iterate through functions (like neurons/filters) and calc max value
# iterate through oi * samples (batches/core elements/weights of neuron) and calc mean value

@torch.jit.script
def rademacher_weight_loss(param, coef):
    n = 10
    accum = 0
    for i in range(n):
        o = torch.randint(0, 1, param.size())
        o[o == 0] = -1.0
        l = []
        for j in range(param.size()[0]):
            d = param[j]
            l.append(torch.sum(o * d) / d.numel())
        accum += torch.max(torch.tensor(l))
    return accum * coef / n


@torch.jit.script
def rademacher_inner_data_loss(param, coef):
    n = 10
    accum = 0
    for i in range(n):

        o = torch.randint(0, 1, param.size())
        o[o == 0] = -1.0
        l = []
        for j in range(param.size()[1]):
            d = param[:, j]
            l.append(torch.sum(o * d) / d.numel())
        accum += torch.max(torch.tensor(l))
    return accum * coef / n

# need backward calced firstly
def calc_grad_abs_mean_weights():
    pass

def calc_mean_weights(model):
    return sum([float(torch.sum(x) / x.numel()) for x in model.parameters()])


# @torch.jit.script
def filter_regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1, wcl2 = coefficients
    res = torch.zeros((1)).to(device)
    for i in range(weights.size()[0]):
        res += l1_l2_loss(weights[i], wcl1, wcl2) / (reduce(lambda a, b: a * b, weights[i].size()))
        res += l1_l2_loss(bias[i], wcl1, wcl2)
        res += l1_l2_loss_biased(norm_coef[i], wcl1, wcl2)
        res += l1_l2_loss(norm_bias[i], wcl1, wcl2)
    return torch.sum(res)


def filter_regularization_loss_from_entropy(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1, wcl2 = coefficients
    res = torch.zeros((1)).to(device)
    for i in range(weights.size()[0]):
        res += l1_l2_loss(weights[i], wcl1, wcl2) / (reduce(lambda a, b: a * b, weights[i].size()))
        res += l1_l2_loss(bias[i], wcl1, wcl2)
        res += l1_l2_loss_biased(norm_coef[i], wcl1, wcl2)
        res += l1_l2_loss(norm_bias[i], wcl1, wcl2)
    return torch.sum(res)


def filter_regularization_loss_from_rademacher(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1, wcl2 = coefficients
    res = torch.zeros((1)).to(device)
    for i in range(weights.size()[0]):
        res += l1_l2_loss(weights[i], wcl1, wcl2) / (reduce(lambda a, b: a * b, weights[i].size()))
        res += l1_l2_loss(bias[i], wcl1, wcl2)
        res += l1_l2_loss_biased(norm_coef[i], wcl1, wcl2)
        res += l1_l2_loss(norm_bias[i], wcl1, wcl2)
    return torch.sum(res)


def filter_regularization_loss_from_grad_(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1, wcl2 = coefficients
    res = torch.zeros((1)).to(device)
    for i in range(weights.size()[0]):
        res += l1_l2_loss(weights[i], wcl1, wcl2) / (reduce(lambda a, b: a * b, weights[i].size()))
        res += l1_l2_loss(bias[i], wcl1, wcl2)
        res += l1_l2_loss_biased(norm_coef[i], wcl1, wcl2)
        res += l1_l2_loss(norm_bias[i], wcl1, wcl2)
    return torch.sum(res)


def block_regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1, wcl2 = coefficients
    res = torch.zeros((1)).to(device)
    res += l1_l2_loss(weights, wcl1, wcl2) / (reduce(lambda a, b: a * b, weights.size()))
    res += l1_l2_loss(bias, wcl1, wcl2) / (reduce(lambda a, b: a * b, bias.size()))
    res += l1_l2_loss_biased(norm_coef, wcl1, wcl2) / (reduce(lambda a, b: a * b, norm_coef.size()))
    res += l1_l2_loss(norm_bias, wcl1, wcl2) / (reduce(lambda a, b: a * b, norm_bias.size()))
    return torch.sum(res)
