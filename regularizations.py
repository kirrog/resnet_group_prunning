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


# @torch.jit.script
def filter_regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, wcl1, wcl2):
    res = torch.zeros((1)).cuda()
    for i in range(weights.size()[0]):
        res += l1_l2_loss(weights[i], wcl1, wcl2) / (reduce(lambda a, b: a * b, weights[i].size()))
        res += l1_l2_loss(bias[i], wcl1, wcl2)
        res += l1_l2_loss_biased(norm_coef[i], wcl1, wcl2)
        res += l1_l2_loss(norm_bias[i], wcl1, wcl2)
    return torch.sum(res)


def calc_mean_weights(model):
    return sum([float(torch.sum(x) / (reduce(lambda a, b: a * b, x.size()))) for x in model.parameters()])


def block_regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, wcl1, wcl2):
    res = torch.zeros((1)).cuda()
    res += l1_l2_loss(weights, wcl1, wcl2) / (reduce(lambda a, b: a * b, weights.size()))
    res += l1_l2_loss(bias, wcl1, wcl2) / (reduce(lambda a, b: a * b, bias.size()))
    res += l1_l2_loss_biased(norm_coef, wcl1, wcl2) / (reduce(lambda a, b: a * b, norm_coef.size()))
    res += l1_l2_loss(norm_bias, wcl1, wcl2) / (reduce(lambda a, b: a * b, norm_bias.size()))
    return torch.sum(res)
