import os
import math
import torch
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def bernoulli_log_pdf(x, probs):
    r"""Log-likelihood of data given ~Bernoulli(mu)
    @param x: PyTorch.Tensor
              ground truth input
    @param mu: PyTorch.Tensor
               Bernoulli distribution parameters
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    return torch.distributions.bernoulli.Bernoulli(probs=probs).log_prob(x)


def masked_bernoulli_log_pdf(x, mask, probs):
    dist = torch.distributions.bernoulli.Bernoulli(probs=probs)
    # log_prob = (1-probs)**0*((1-x)*0.86 + 0.14 * x) * dist.log_prob(x.relu())
    log_prob = dist.log_prob(x.relu())

    return log_prob * mask.float()



def masked_gaussian_log_pdf(x, mask, mu, logvar):
    sigma = torch.exp(0.5 * logvar)
    dist = torch.distributions.normal.Normal(mu, sigma)
    log_prob = dist.log_prob(x)
    return log_prob * mask.float()


def normal_log_pdf(x, mu, logvar):
    scale = torch.exp(0.5 * logvar)
    return torch.distributions.normal.Normal(mu, scale).log_prob(x)


def standard_normal_log_pdf(x):
    mu = torch.zeros_like(x)
    scale = torch.ones_like(x)
    return torch.distributions.normal.Normal(mu, scale).log_prob(x)


def log_mean_exp(x, dim=1):
    """log(1/k * sum(exp(x))): this normalizes x.
    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    # m = torch.max(x, dim=dim, keepdim=True)[0]
    # return m + torch.log(torch.mean(torch.exp(x - m),
                        #  dim=dim, keepdim=True))
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[1])


def kl_divergence_standard_normal_prior(z_mu, z_logvar):
    kl_div = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_div = torch.sum(kl_div, dim=1)
    return kl_div


def kl_divergence_normal_prior(q_z_mu, q_z_logvar, p_z_mu, p_z_logvar):
    q = dist.Normal(q_z_mu, torch.exp(0.5 * q_z_logvar))
    p = dist.Normal(p_z_mu, torch.exp(0.5 * p_z_logvar))
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def gentr_fn(alist):
    while 1:
        for j in alist:
            yield j


def product_of_experts(mu, logvar, eps=1e-8):
    # assume the first dimension is the number of experts
    var = torch.exp(logvar) + eps
    T = 1 / var  # precision of i-th Gaussian expert at point x
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1 / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var)

    return pd_mu, pd_logvar


def multivariate_product_of_experts(mu, logcov, eps=1e-8):
    # assume the first dimension is the number of experts
    # we also assume logcov is already in square form
    cov = torch.exp(logcov) + eps
    T = torch.inverse(cov)
    sum_T = torch.sum(T, dim=0)
    sum_T_inv = torch.inverse(T_cov)

    mT = torch.sum(torch.einsum('pbi,pbii->pbi', mu, T), dim=0)
    pd_mu = torch.einsum('bii,bii->bii', mT, sum_T_inv)
    
    pd_cov = sum_T_inv
    pd_logcov = torch.log(pd_cov + eps)

    return pd_mu, pd_logcov

def compute_auroc(actual, predicted):
    """
    Computes the area under the receiver-operator characteristic curve.
    This code a rewriting of code by Ben Hamner, available here:
    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    num = len(actual)
    temp = sorted([[predicted[i], actual[i]] for i in range(num)], reverse=True)

    sorted_predicted = [row[0] for row in temp]
    sorted_actual = [row[1] for row in temp]

    sorted_posterior = sorted(zip(sorted_predicted, range(len(sorted_predicted))))
    r = [0 for k in sorted_predicted]
    cur_val = sorted_posterior[0][0]
    last_rank = 0
    for i in range(len(sorted_posterior)):
        if cur_val != sorted_posterior[i][0]:
            cur_val = sorted_posterior[i][0]
            for j in range(last_rank, i):
                r[sorted_posterior[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_posterior)-1:
            for j in range(last_rank, i+1):
                r[sorted_posterior[j][1]] = float(last_rank+i+2)/2.0

    num_positive = len([0 for x in sorted_actual if x == 1])
    num_negative = num - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if sorted_actual[i] == 1])
    auroc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive))

    return auroc

def evaluate_metrics(actual, predicted):
    """
    This computes and returns a dictionary of notable evaluation metrics for your predicted labels.
    """
    acc = compute_acc(actual, predicted)
    auroc = compute_auroc(actual, predicted)
    F1 = compute_f1(actual, predicted)

    return {'accuracy': acc, 'auroc': auroc, 'F1': F1}


def compute_f1(actual, predicted):
    """
    Computes the F1 score of your predictions. Note that we use 0.5 as the cutoff here.
    """
    num = len(actual)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for i in range(num):
        if actual[i] >= 0.5 and predicted[i] >= 0.5:
            true_positives += 1
        elif actual[i] < 0.5 and predicted[i] >= 0.5:
            false_positives += 1
        elif actual[i] >= 0.5 and predicted[i] < 0.5:
            false_negatives += 1
        else:
            true_negatives += 1

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        F1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        F1 = 0.0

    return F1

def compute_acc(actual, predicted):
    """
    Computes the accuracy of your predictions, using 0.5 as a cutoff.

    Note that these inputs are lists, not dicts; they assume that actual and predicted are in the same order.

    Parameters (here and below):
        actual: a list of the actual labels
        predicted: a list of your predicted labels
    """
    num = len(actual)
    acc = 0.
    for i in range(num):
        if round(actual[i], 0) == round(predicted[i], 0):
            acc += 1.
    acc /= num
    return acc