import numpy as np
import tensorflow as tf

class MultiLayerPerceptron(object):

    def __init__(self, input_tensor, layers, activation=tf.nn.relu,
                 regul = tf.contrib.layers.l2_regularizer(0.1)):
        """Sets up a multilayer perceptron computation graph."""
        with input_tensor.graph.as_default():
            transform = input_tensor
            for units in layers:
                transform = tf.contrib.layers.fully_connected(
                    transform, units, weights_regularizer=regul,
                    activation_fn=activation)
        self.output = transform

    def get_output_layer(self):
        """Retursn output tensor of the multilayer perceptron."""
        return self.output

class ConditionalStochasticModel(object):
    """Abstract class for conditional probability models P(A|B)."""

    def __init__(self, a, b, has_analytical_entropy=False):
        """Sets up the basic properties of random variable."""
        if not a.graph == b.graph:
            raise ValueError(
                'Tensor a does not share computation graph with tensor b.')
        self.graph = a.graph
        self.has_analytical_entropy = has_analytical_entropy

    def log_density(self):
        """Computes log density of latent code."""
        pass

    def sample(self, n_samples=1):
        """Samples n latent code from the variational family."""
        pass

    def expected_value(self, function):
        """Computes expected value of function """
        pass

    def entropy(self):
        """Computes (estimate or analytical) entropy of the variational distribution."""
        pass

class MultiLayerStochastic(ConditionalStochasticModel):
    """Implements multilayer perceptron conditional normal."""

    def __init__(self, a, b, layers):
        super(MultiLayerStochastic, self).__init__(a, b)
        # Reparametrization of the random variable.
        dim = a.shape[1]
        with self.graph.as_default():
            self.noise = tf.distributions.Normal(
                loc=np.zeros(dim), scale=np.ones(dim))
        self.mu = MultiLayerPerceptron(b, layers).get_output_layer()
        self.sigma = MultiLayerPerceptron(b, layers).get_output_layer()
        self.distribution = tf.distributions.Normal(
            loc=self.mu, scale=self.sigma)

    def log_density(self):
        return self.distribution.prob(self.a)

    def sample(self, n_samples=1):
        with self.graph.as_default():
            noise_samples = self.noise.sample(n_samples)
            reparam = noise_samples + self.mu * self.sigma
            return reparam

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
        pass

    def fit(self, n_batches, n_epochs):
        """Fit the recognition model and data model simultaneously."""
        pass

    def decode(self, latent_codes):
        """Reconstructs observations using the given codes."""
        pass

    def encode(self, observations):
        """Probabilistic code of an observation."""
        pass

    def current_loss(self):
        pass

class NormalizingFlow(ConditionalStochasticModel):
    """Normalizing flow random variable for hooshmand shokri razaghi."""

    def __init__(self, stochastic):
        super(NormalizingFlow, self).__init__(graph)


