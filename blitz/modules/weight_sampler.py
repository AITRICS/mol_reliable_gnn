import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

class GaussianVariational(nn.Module):
    #Samples weights for variational inference as in Weights Uncertainity on Neural Networks (Bayes by backprop paper)
    #Calculates the variational posterior part of the complexity part of the loss
    def __init__(self, mu, rho):
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.w = None
        self.sigma = None
        self.pi = np.pi
        self.normal = torch.distributions.Normal(0, 1)

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is 
        a function from a trainable parameter, and adding a mean

        sets those weights as the current ones

        returns:
            torch.tensor with same shape as self.mu and self.rho
        """
        device = self.mu.device
        epsilon = self.normal.sample(self.mu.size()).to(device)
        self.sigma = torch.log(1 + torch.exp(self.rho)).to(device)
        self.w = self.mu + self.sigma * epsilon
        return self.w

    def log_posterior(self):

        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """

        assert (self.w is not None), "You can only have a log posterior for W if you've already sampled it"

        log_sqrt2pi = np.log(np.sqrt(2*self.pi))
        log_posteriors =  -log_sqrt2pi - self.sigma - (((self.w - self.mu) ** 2)/(2 * self.sigma ** 2))
        return log_posteriors.mean()

class ScaleMixturePrior(nn.Module):
    #Calculates a Scale Mixture Prior distribution for the prior part of the complexity cost on Bayes by Backprop paper
    def __init__(self, pi, sigma1, sigma2, dist):
        super().__init__()

        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.normal1 = torch.distributions.Normal(0, sigma1)
        self.normal2 = torch.distributions.Normal(0, sigma2)

    def log_prior(self, w):
        """
        Calculates the log_likelihood for each of the weights sampled relative to a prior distribution as a part of the complexity cost

        returns:
            torch.tensor with shape []
        """
        prob_n1 = torch.exp(self.normal1.log_prob(w))
        prob_n2 = torch.exp(self.normal2.log_prob(w))
        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2)

        return (torch.log(prior_pdf)).mean()
