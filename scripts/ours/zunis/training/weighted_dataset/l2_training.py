"""Optimization of invertible flows in the weighted dataset problem using the DKL loss

Reminder: we have a dataset (x,p(x),f(x)) such that
- x ~ p(x)
- we want to learn a model that draws points according to f(x),
which is positive, and known up to normalization

We want to optimize a function q(x) such that doing importance sampling to compute f(x)
with it minimizes the variance.

The variance of the importance sampling estimator is our proto-loss

pL(f,q) =  ∫ dx q(x) (f(x)/q(x))^2 - (∫ dx q(x) f(x)/q(x))^2
       =  ∫ dx (f(x)^2/q(x)) - I(f)^2

where I(f) is the integral we want to compute and is independent of q, so our real loss is

L(f,q) = ∫ dx f(x)^2/q(x)

Which we further can compute using importance sampling from p(x):

L(f,q) = ∫ dx p(x) f(x)^2/q(x)/p(x)

Which we can compute from our dataset as the expectation value

L(f,q) = E(f(x)^2/(q(x) p(x)), x~p(x)
"""

import logging
import torch
from .weighted_dataset_trainer import BasicTrainer, BasicStatefulTrainer

logger = logging.getLogger(__name__)

def weighted_l2_loss(fx, px, logqx, div=None, est=None):

    pred = torch.exp(logqx)
    if est is not None:
        pred = pred*est

    pred = pred/px


    rL2 = (px*((fx -pred))**2)
    if div is not None:
        rL2 = rL2/div

    rL2 = rL2[~(torch.isinf(rL2) | torch.isnan(rL2))]
    rL2 = rL2.mean()
    return rL2

class BasicL2Trainer(BasicTrainer):
    """Basic trainer based on the variance loss"""
    def __init__(self, flow, latent_prior):
        super(BasicL2Trainer, self).__init__(flow, latent_prior)
        self.loss = weighted_l2_loss


class BasicStatefulL2Trainer(BasicStatefulTrainer):
    """Basic stateful trainer based on the variance loss"""
    def __init__(self, flow, latent_prior, **kwargs):
        super(BasicStatefulL2Trainer, self).__init__(flow, latent_prior, **kwargs)
        self.loss = weighted_l2_loss
