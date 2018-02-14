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
    """Abstract class for conditional probability models P(X|A)."""

    def __init__(self, dim, a):
        """Sets up the basic properties of random variable.

        Args:
        dim: int
            Dimension of X (random variable).
        a: tf.Tensor
            Random variable A that X has been conditioned upon.
        """
        if not isinstance(a, tf.Tensor):
            raise ValueError(
                'Random variable A should be a tensor.')
        self.graph = a.graph
        self.dim = dim

    def log_density(self, x):
        """Computes log density of latent code.

        Parameters:
        -----------
        x: tf.Tensor of shape [None, self.dim]
        """
        pass

    def neg_log_density(self, x):
        """Computes negative log density of latent code.

        Parameters:
        -----------
        x: tf.Tensor of shape [None, self.dim]
        """
        return - self.log_density(x)

    def cond_sample(self, n_samples=1):
        """Samples n latent code from the variational family.

        should return samples of random variable a given b value.
        """
        pass

    def expected_value(self, function, n_samples=1):
        """Computes expected value of function using Monte Carlo estimation.

        function: Tensorflow op.
        """
        return tf.reduce_mean(function(self.cond_sample(n_samples)), axis=0)

    def entropy(self, n_samples=1):
        """Computes monte-carlo estimate of entropy of the random variable."""
        return self.expected_value(self.neg_log_density, n_samples=n_samples)


class MultiLayerStochastic(ConditionalStochasticModel):
    """Implements multilayer perceptron conditional normal."""

    def __init__(self, dim, a, hidden_layers):
        super(MultiLayerStochastic, self).__init__(dim, a)
        # Reparametrization of the random variable.
        self.a = a
        layers = hidden_layers + [self.dim]
        with self.graph.as_default():
            self.mu = MultiLayerPerceptron(self.a, layers).get_output_layer()
            self.sigma = MultiLayerPerceptron(
                self.a, layers).get_output_layer()
            # Added epsilon term basically initializes scale to be non-zero.
            epsilon = 0.000001
            self.scale = tf.pow(self.sigma, 2) + epsilon
            self.distribution = tf.distributions.Normal(
                loc=self.mu, scale=self.scale)

    def log_density(self, x):
        return tf.reduce_sum(self.distribution.log_prob(x), axis=-1)

    def cond_sample(self, n_samples=1):
        with self.graph.as_default():
            samples = self.distribution.sample(n_samples)
            return samples


class AutoEncodingVariationalBayes(object):

    def __init__(self, prob_model, recognition_model, prior, optimizer):
        """Sets up the data/recognition model and the optimizer.

        Parameters:
        -----------
        prob_model: ConditionalStochasticModel
            Likelihood model P(X|Z)
        recognition_model: ConditionalStochasticModel
            Variational model Q(Z|X)
        """
        if not isinstance(prob_model, ConditionalStochasticModel):
            raise ValueError(
                "prob_model should be of type 'ConditionalStochasticModel'.")
        if not isinstance(recognition_model, ConditionalStochasticModel):
            raise ValueError(
                "recognition_model should be of type 'ConditionalStochasticModel'.")
        if not recognition_model.graph == prob_model.graph:
            raise ValueError(
                "Both models should share the same graph.")
        if not isinstance(optimizer, tf.train.Optimizer):
            raise ValueError("optimizer should be of type 'tf.train.Optimizer'.")
        # Probability model plays the role of decoder
        # and recognition model plays the role of encoder.
        self.graph = prob_model.graph
        self.encoder = recognition_model
        self.decoder = prob_model
        self.x = self.encoder.a
        self.z = self.decoder.a
        # Prior
        self.prior = prior
        self.session = None
        # Optimizer
        self.optimizer = optimizer
        self.set_up_divergence_loss()

    def set_up_divergence_loss(self):
        """Divergence metric for comparing posterior and variational family.

        If there is analytical divergence between the models
        or analytical entropy of the recognition model, substitute.
        """
        # We'll set up the ELBO loss.
        with self.graph.as_default():
            self.loss = 0
            # Expected value of Log-likelihood with respect to
            # recognition model.
            self.loss += tf.reduce_sum(
                tf.reduce_mean(self.decoder.log_density(self.x), axis=0))
            # Entropy of recognition model.
            self.loss += tf.reduce_sum(
                self.encoder.entropy(n_samples=self.z.shape[0].value))
            # Expected value of the prior on latent space.
            self.loss += tf.reduce_sum(
                self.prior.log_prob(self.z))
            self.train_op = self.optimizer.minimize(-self.loss)
            self.recon = self.decoder.cond_sample()

    def initialize(self):
        """Initializes the variables of the computation graph.

        Note that the computation graph here consists of the
        recognition model, likelihood model and the variational
        lower bound of the two. Every time this method is called,
        the entire set of variables are reset.
        """
        with self.graph.as_default():
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    def fit(self, observations):
        """Fit the recognition model and data model simultaneously one step."""
        with self.graph.as_default():
            return self.session.run(
                [self.train_op, self.loss], feed_dict={self.x: observations})

    def code_reconstruct(self, observations):
        """Reconstructs observations using the given codes."""
        with self.graph.as_default():
            return self.session.run(
                [self.z, self.recon], feed_dict={self.x: observations})

    def loss(self, observations):
        """Computes the current evidence lower-bound."""
        with self.graph.as_default():
            return self.session.run(
                self.loss, feed_dict={self.x: observations})

class NormalizingFlow(ConditionalStochasticModel):
    """Normalizing flow random variable for hooshmand shokri razaghi."""

    def __init__(self, graph, dim):
        super(NormalizingFlow, self).__init__(graph, dim)
