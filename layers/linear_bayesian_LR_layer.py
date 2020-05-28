import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import GaussianVariational, ScaleMixturePrior

def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD

class BayesianLinear(BayesianModule):
    """
    Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers
    
    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not
    
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.4,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -6.0,
                 freeze = False,
                 prior_dist = None):
        super().__init__()

        #our main parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        # Variational weight parameters and sample
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = GaussianVariational(self.weight_mu, self.weight_rho)

        # Variational bias parameters and sample
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1))
            self.bias_sampler = GaussianVariational(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        if self.bias:
            self.bias_prior_dist = ScaleMixturePrior(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # Sample the weights and forward it
        
        #if the model is frozen, return frozen
        if self.freeze:
            return self.forward_frozen(x)

        #   Calculate stf
        std_w = 1e-6 + F.softplus(self.weight_sampler.rho, beta=1, threshold=20)
        act_w_mu = F.linear(x, self.weight_sampler.mu)
        act_w_std = torch.sqrt(F.linear(x.pow(2), std_w.pow(2)))
        eps_w = Variable(self.weight_sampler.mu.data.new(act_w_std.size()).normal_(mean=0, std=1))

        if self.bias:
            std_b = 1e-6 + F.softplus(self.bias_sampler.rho, beta=1, threshold=20)
            eps_b = Variable(self.bias_sampler.mu.data.new(std_b.size()).normal_(mean=0, std=0.1))

        act_w_out = act_w_mu + act_w_std * eps_w
        if self.bias:
            act_b_out = self.bias_sampler.mu + std_b * eps_b

        if self.bias:
            output = act_w_out + act_b_out.unsqueeze(0).expand(x.shape[0], -1)
        else:
            output = act_w_out

        self.log_variational_posterior =  KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.weight_sampler.mu, sig_q=std_w)
        if self.bias:
            self.log_variational_posterior +=  KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.bias_sampler.mu, sig_q=std_b)
        '''
        self.weight_sampler.w = act_w_out
        self.log_variational_posterior = self.weight_sampler.log_posterior()
        self.log_prior = self.weight_prior_dist.log_prior(act_w_out)
        if self.bias:
            self.bias_sampler.w = act_b_out
            self.log_variational_posterior += self.bias_sampler.log_posterior()
            self.log_prior += self.bias_prior_dist.log_prior(act_b_out)
        '''

        return output

    def forward_frozen(self, x):
        """
        Computes the feedforward operation with the expected value for weight and biases
        """
        if self.bias:
            return F.linear(x, self.weight_mu, self.bias_mu)
        else:
            return F.linear(x, self.weight_mu, torch.zeros(self.out_features))
