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
    # d[d == 0.0] = coef
    d += coef
    entr = torch.sum(torch.abs(- d * torch.log(d) * coef)) / d.numel()
    return entr


def calc_mean_weights(model):
    return sum([float(torch.sum(x) / x.numel()) for x in model.parameters()])


# @torch.jit.script
def filter_regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1, wcl2 = coefficients
    res = torch.zeros((1)).to(device)
    for i in range(weights.size()[0]):
        res += l1_l2_loss(weights[i], wcl1, wcl2) / weights[i].numel()
        res += l1_l2_loss(bias[i], wcl1, wcl2)
        res += l1_l2_loss_biased(norm_coef[i], wcl1, wcl2)
        res += l1_l2_loss(norm_bias[i], wcl1, wcl2)
    return torch.sum(res)


def filter_regularization_loss_from_entropy(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1 = coefficients[0]
    res = torch.zeros((1)).to(device)
    for i in range(weights.size()[0]):
        res += entropy_loss(weights[i], wcl1)
        res += entropy_loss(bias[i], wcl1)
        res += entropy_loss(norm_coef[i], wcl1)
        res += entropy_loss(norm_bias[i], wcl1)
    return torch.sum(res)


def filter_regularization_loss_from_entropy_inv(weights, bias, norm_coef, norm_bias, coefficients, device):
    return -filter_regularization_loss_from_entropy(weights, bias, norm_coef, norm_bias, coefficients, device)


# iterate by version of oi and calc mean - mat ojid
# iterate through functions (like neurons/filters) and calc max value
# iterate through oi * samples (batches/core elements/weights of neuron) and calc mean value

@torch.jit.script
def rademacher_weight_loss(param, coef, o):
    res = o * param
    if len(res.size()) > 2:
        s = torch.sum(res, dim=list(range(len(res.size())))[2:]) / param[0].numel()
    else:
        s = res
    if len(s.size()) > 1:
        m = torch.max(s, dim=1)[0]
    else:
        m = s
    return torch.mean(m) * coef


def filter_regularization_loss_from_rademacher(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1 = coefficients[0]
    res = torch.zeros((1)).to(device)
    n = 10

    weights_o = torch.rand([n] + list(weights[0].size()), device=device)
    weights_o[weights_o >= 0.5] = 1.0
    weights_o[weights_o <= 0.5] = -1.0
    bias_o = torch.rand([n] + list(bias[0].size()), device=device)
    bias_o[bias_o >= 0.5] = 1.0
    bias_o[bias_o <= 0.5] = -1.0
    norm_coef_o = torch.rand([n] + list(norm_coef[0].size()), device=device)
    norm_coef_o[norm_coef_o >= 0.5] = 1.0
    norm_coef_o[norm_coef_o <= 0.5] = -1.0
    norm_bias_o = torch.rand([n] + list(norm_bias[0].size()), device=device)
    norm_bias_o[norm_bias_o >= 0.5] = 1.0
    norm_bias_o[norm_bias_o <= 0.5] = -1.0
    for i in range(weights.size()[0]):
        res += rademacher_weight_loss(weights[i], wcl1, weights_o)
        res += rademacher_weight_loss(bias[i], wcl1, bias_o)
        res += rademacher_weight_loss(norm_coef[i], wcl1, norm_coef_o)
        res += rademacher_weight_loss(norm_bias[i], wcl1, norm_bias_o)
    return torch.sum(res)


def filter_regularization_loss_from_rademacher_inv(weights, bias, norm_coef, norm_bias, coefficients, device):
    return - filter_regularization_loss_from_rademacher(weights, bias, norm_coef, norm_bias, coefficients, device)


def block_regularization_loss_from_weights(weights, bias, norm_coef, norm_bias, coefficients, device):
    wcl1, wcl2 = coefficients
    res = torch.zeros((1)).to(device)
    res += l1_l2_loss(weights, wcl1, wcl2) / weights.numel()
    res += l1_l2_loss(bias, wcl1, wcl2) / bias.numel()
    res += l1_l2_loss_biased(norm_coef, wcl1, wcl2) / norm_coef.numel()
    res += l1_l2_loss(norm_bias, wcl1, wcl2) / norm_bias.numel()
    return torch.sum(res)


# For prunning score calculation
def filter_regularization_feature_from_weights(weights, bias, norm_coef, norm_bias, out_weights, device):
    wcl1, wcl2 = [torch.as_tensor(x).to(device) for x in (1e-10, 1e-9)]
    res = torch.zeros(weights.size(0)).to(device)
    for i in range(weights.size(0)):
        res[i] += l1_l2_loss(weights[i], wcl1, wcl2) / weights[i].numel()
        res[i] += l1_l2_loss(bias[i], wcl1, wcl2)
        res[i] += l1_l2_loss_biased(norm_coef[i], wcl1, wcl2)
        res[i] += l1_l2_loss(norm_bias[i], wcl1, wcl2)
        res[i] += l1_l2_loss(out_weights[:, i], wcl1, wcl2) / out_weights[:, i].numel()
    return res


def filter_regularization_feature_from_weights_inv(weights, bias, norm_coef, norm_bias, out_weights, device):
    return -filter_regularization_feature_from_weights(weights, bias, norm_coef, norm_bias, out_weights, device)


def filter_regularization_feature_from_entropy(weights, bias, norm_coef, norm_bias, out_weights, device):
    wcl1 = torch.as_tensor(1e-10).to(device)
    res = torch.zeros(weights.size(0)).to(device)
    for i in range(weights.size(0)):
        res[i] += entropy_loss(weights[i], wcl1)
        res[i] += entropy_loss(bias[i], wcl1)
        res[i] += entropy_loss(norm_coef[i], wcl1)
        res[i] += entropy_loss(norm_bias[i], wcl1)
        res += entropy_loss(out_weights[:, i], wcl1)
    return res


def filter_regularization_feature_from_entropy_inv(weights, bias, norm_coef, norm_bias, out_weights, device):
    return -filter_regularization_feature_from_entropy(weights, bias, norm_coef, norm_bias, out_weights, device)


def filter_regularization_feature_from_rademacher(weights, bias, norm_coef, norm_bias, out_weights, device):
    wcl1 = torch.as_tensor(1e-10).to(device)
    n = 10

    weights_o = torch.rand([n] + list(weights[0].size()), device=device)
    weights_o[weights_o >= 0.5] = 1.0
    weights_o[weights_o <= 0.5] = -1.0
    bias_o = torch.rand([n] + list(bias[0].size()), device=device)
    bias_o[bias_o >= 0.5] = 1.0
    bias_o[bias_o <= 0.5] = -1.0
    norm_coef_o = torch.rand([n] + list(norm_coef[0].size()), device=device)
    norm_coef_o[norm_coef_o >= 0.5] = 1.0
    norm_coef_o[norm_coef_o <= 0.5] = -1.0
    norm_bias_o = torch.rand([n] + list(norm_bias[0].size()), device=device)
    norm_bias_o[norm_bias_o >= 0.5] = 1.0
    norm_bias_o[norm_bias_o <= 0.5] = -1.0
    out_weights_o = torch.rand([n] + list(out_weights[:, 0].size()), device=device)
    out_weights_o[out_weights_o >= 0.5] = 1.0
    out_weights_o[out_weights_o <= 0.5] = -1.0

    res = torch.zeros(weights.size(0)).to(device)
    for i in range(weights.size(0)):
        res[i] += rademacher_weight_loss(weights[i], wcl1, weights_o)
        res[i] += rademacher_weight_loss(bias[i], wcl1, bias_o)
        res[i] += rademacher_weight_loss(norm_coef[i], wcl1, norm_coef_o)
        res[i] += rademacher_weight_loss(norm_bias[i], wcl1, norm_bias_o)
        res[i] += rademacher_weight_loss(out_weights[:, i], wcl1, out_weights_o)
    return res


def filter_regularization_feature_from_rademacher_inv(weights, bias, norm_coef, norm_bias, out_weights, device):
    return - filter_regularization_feature_from_rademacher(weights, bias, norm_coef, norm_bias, out_weights, device)
