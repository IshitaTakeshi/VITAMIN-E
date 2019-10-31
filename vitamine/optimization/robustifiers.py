from autograd import elementwise_grad
from autograd import numpy as np


# Eade, Ethan. "Gauss-newton/levenberg-marquardt optimization." Tech. Rep. (2013).
# http://ethaneade.com/optimization.pdf


class BaseRobustifier(object):
    def robustify(self, x):
        raise NotImplementedError()

    def grad(self, x):
        return elementwise_grad(self.robustify)(x)

    def weights(self, x):
        mask = (x != 0)

        # process only nonzero members to avoid division by zero
        # for members where x == 0, we set 0 to corresponding y
        # because usually x = norm(residual), so x == 0 means that
        # residual is zero and weighting is not required

        y = np.zeros(x.shape)
        y[mask] = self.grad(x[mask]) / x[mask]
        return y


class SquaredRobustifier(BaseRobustifier):
    def robustify(self, x):
        return np.power(x, 2)


def huber_(x, k):
    k2 = k * k
    if x < k2:
        return x
    return 2 * k * np.sqrt(x) - k2


class HuberRobustifier(BaseRobustifier):
    def __init__(self, k):
        assert(k > 0)
        self.k = k

    def robustify(self, errors):
        # Take weigthed quadratic error such that
        # errors[i] = e_i.T * W * e_i, e_i = (x_pred[i] - x_true[i])
        assert(np.all(errors >= 0.0))
        # we need to do this to calculate grad
        return np.array([huber_(v, self.k) for v in errors])


class GemanMcClureRobustifier(BaseRobustifier):
    def __init__(self, sigma=0.1):
        self.v = np.power(sigma, 2)

    def robustify(self, x):
        u = np.power(x, 2)
        return u / (u + self.v)
