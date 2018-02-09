import tensorflow as tf


class ConditionalDistribution(object):
    """Class for encapsulation of recognition model"""

    def __init__(self, graph):
        """Sets up the basic properties of random variable."""
        self.graph = graph
        self.has_analytical_entropy = False

    def log_density(self, latent):
        """Computes log density of latent code."""
        pass

    def sample(self, n_samples):
        """Samples n latent code from the variational family."""
        pass

    def entropy(self):
        """Computes (estimate or analytical) entropy of the variational distribution."""
        pass

    def expected_value(self, function):
        """Computes expected value of function """
        pass


class AutoEncodingVariationalBayes(object):

    def __init__(self, prob_model, recognition_model, optimizer):
        """Sets up the data/recognition model and the optimizer."""
        self.encoder = recognition_model
        self.decoder = prob_model
        self.optimizer = optimizer

    def divergence_loss(self, observation):
        """Divergence metric for comparing posterior and variational family.

        If there is analytical divergence between the models
        or analytical entropy of the recognition model, substitute.
        """
        if 

    def fit(self, n_batches, n_epochs):
        """Fit the recognition model and data model simultaneously."""
        pass

    def decode(self, latent_codes):
        """Reconstructs observations using the given codes."""
        pass

    def encode():
        """Probabilistic code of an observation."""
        pass

    def current_loss():
        pass

class NormalizingFlow(RandomVariable):
    """Normalizing flow random variable for hooshmand shokri razaghi."""

    def __init__(self, graph):
        super(NormalizingFlow, self).__init__(graph)


class MultiLayerPerceptronRandomVariable(RandomVariable):

    def __init__(self, graph):
        super(MultiLayerPerceptronRandomVariable, self).__init__(graph)
