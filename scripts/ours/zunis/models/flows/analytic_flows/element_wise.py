from .analytic_flow import AnalyticFlow,InvertibleAnalyticFlow
import torch


class AnalyticSigmoid(AnalyticFlow):
    def __init__(self,d):
        super(AnalyticSigmoid, self).__init__(d=d)
        self.flow_ = torch.nn.Sigmoid()

    def transform_and_compute_jacobian(self, xj, opt_feats=None):
        x = xj[..., :-1]
        logj = xj[..., -1]
        yj = torch.zeros_like(xj).to(xj.device)
        yj[..., :-1] = self.flow(x, opt_feats)

        yj[..., -1] = logj+torch.sum(torch.log(yj[..., :-1] * (1 - yj[..., :-1])), dim=-1)

        return yj


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class AnalyticInverseSigmoid(AnalyticFlow):
    def __init__(self,d):
        super(AnalyticInverseSigmoid, self).__init__(d=d)
        self.flow_ = inverse_sigmoid

    def transform_and_compute_jacobian(self, xj, opt_feats=None):
        x = xj[..., :-1]
        logj = xj[..., -1]
        yj = torch.zeros_like(xj).to(xj.device)
        yj[..., :-1] = self.flow(x, opt_feats)

        yj[..., -1] = logj-torch.sum(torch.log(x * (1 - x)), dim=-1)

        return yj


class InvertibleAnalyticSigmoid(InvertibleAnalyticFlow):
    def __init__(self,d):
        super(InvertibleAnalyticSigmoid, self).__init__(d=d)
        self.forward_flow = AnalyticSigmoid(d)
        self.backward_flow = AnalyticInverseSigmoid(d)
