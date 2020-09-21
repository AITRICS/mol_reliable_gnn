import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from blitz.losses import kl_divergence_from_nn
from blitz.modules import BayesianLinear, BayesianConv2d

class TestKLDivergence(unittest.TestCase):

    def test_kl_divergence_bayesian_linear_module(self):
        blinear = BayesianLinear(10, 10)
        to_feed = torch.ones((1, 10))
        predicted = blinear(to_feed)

        complexity_cost = blinear.log_variational_posterior - blinear.log_prior
        kl_complexity_cost = kl_divergence_from_nn(blinear)

        self.assertEqual((complexity_cost == kl_complexity_cost).all(), torch.tensor(True))
        pass
    
    def test_kl_divergence_bayesian_conv2d_module(self):
        bconv = BayesianConv2d(in_channels=3,
                              out_channels=3,
                              kernel_size=(3,3))

        to_feed = torch.ones((1, 3, 25, 25))
        predicted = bconv(to_feed)
        
        complexity_cost = bconv.log_variational_posterior - bconv.log_prior
        kl_complexity_cost = kl_divergence_from_nn(bconv)

        self.assertEqual((complexity_cost == kl_complexity_cost).all(), torch.tensor(True))
        pass

    def test_kl_divergence_non_bayesian_module(self):
        linear = nn.Linear(10, 10)
        to_feed = torch.ones((1, 10))
        predicted = linear(to_feed)

        kl_complexity_cost = kl_divergence_from_nn(linear)
        self.assertEqual((torch.tensor(0) == kl_complexity_cost).all(), torch.tensor(True))
        pass

if __name__ == "__main__":
    unittest.main()