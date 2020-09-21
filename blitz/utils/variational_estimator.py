import torch
from blitz.losses import kl_divergence_from_nn
from blitz.modules.base_bayesian_module import BayesianModule

def variational_estimator(nn_class):
    """
    This decorator adds some util methods to a nn.Module, in order to facilitate the handling of Bayesian Deep Learning features

    Parameters:
        nn_class: torch.nn.Module -> Torch neural network module

    Returns a nn.Module with methods for:
        (1) Gathering the KL Divergence along its BayesianModules;
        (2) Sample the Elbo Loss along its variational inferences (helps training)
        (3) Freeze the model, in order to predict using only their weight distribution means
    """

    def nn_kl_divergence(self):
        """Returns the sum of the KL divergence of each of the BayesianModules of the model, which are from
            their posterior current distribution of weights relative to a scale-mixtured prior (and simpler) distribution of weights

            Parameters:
                N/a

            Returns torch.tensor with 0 dim.      
        
        """
        return kl_divergence_from_nn(self)
    
    setattr(nn_class, "nn_kl_divergence", nn_kl_divergence)

    def sample_elbo(self,
                    inputs,
                    labels,
                    criterion,
                    sample_nbr,
                    complexity_cost_weight=1):

        """ Samples the ELBO Loss for a batch of data, consisting of inputs and corresponding-by-index labels

                The ELBO Loss consists of the sum of the KL Divergence of the model 
                 (explained above, interpreted as a "complexity part" of the loss)
                 with the actual criterion - (loss function) of optimization of our model
                 (the performance part of the loss). 

                As we are using variational inference, it takes several (quantified by the parameter sample_nbr) Monte-Carlo
                 samples of the weights in order to gather a better approximation for the loss.

            Parameters:
                inputs: torch.tensor -> the input data to the model
                labels: torch.tensor -> label data for the performance-part of the loss calculation
                        The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
                criterion: torch.nn.Module, custom criterion (loss) function, torch.nn.functional function -> criterion to gather
                            the performance cost for the model
                sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to 
                            gather the loss to be .backwarded in the optimization of the model.        
        
        """

        loss = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss += self.nn_kl_divergence() * complexity_cost_weight
        return loss / sample_nbr
    
    setattr(nn_class, "sample_elbo", sample_elbo)


    def freeze_model(self):
        """
        Freezes the model by making it predict using only the expected value to their BayesianModules' weights distributions
        """
        for module in self.modules():
            if isinstance(module, (BayesianModule)):
                module.freeze = True

    setattr(nn_class, "freeze", freeze_model)

    def unfreeze_model(self):
        """
        Unfreezes the model by letting it draw its weights with uncertanity from their correspondent distributions
        """
        
        for module in self.modules():
            if isinstance(module, (BayesianModule)):
                module.freeze = False

    setattr(nn_class, "unfreeze", unfreeze_model)
    return nn_class