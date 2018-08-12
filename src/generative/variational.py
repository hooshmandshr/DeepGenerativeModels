import numpy as np
import tensorflow as tf

class MultiLayerPerceptron(object):

    def __init__(self, input_tensor, layers, activation=tf.nn.relu,
                 output_activation=None,
                 regul=tf.contrib.layers.l2_regularizer(0.1)):
        """Sets up a multilayer perceptron computation graph."""
        with input_tensor.graph.as_default():
            transform = input_tensor
            for units in layers:
                transform = tf.contrib.layers.fully_connected(
                    transform, units, weights_regularizer=regul,
                    activation_fn=activation)
        self.output = tf.contrib.layers.fully_connected(
            transform, units, weights_regularizer=regul,
            activation_fn=output_activation)

    def get_output_layer(self):
        """Retursn output tensor of the multilayer perceptron."""
        return self.output


class LogitNormal(tf.distributions.Normal):

    def __init__(self, loc, scale):
        super(LogitNormal, self).__init__(loc=loc, scale=scale)

    def log_prob(self, value, name='log_prob'):
        logit_value = tf.log(value / (1 - value))
        log_prob = super(LogitNormal, self).log_prob(logit_value, name=name)
        return log_prob - tf.log(value) - tf.log(1 - value)

    def sample(self, sample_shape=(), seed=None, name='sample'):
            sample = super(LogitNormal, self).sample(
                sample_shape=sample_shape, seed=seed, name=name)
            return tf.sigmoid(sample)


class ReparameterizedDistribution(object):
    """Reparameterized Distributions for AEVB."""

    def __init__(self, distribution, neural_net_function, **kwargs):
        """Neural net functions to the parameters of the given dist.

        Parameters:
        -----------
        distribution: tensorflow.distributions.Distribution
            distribution that is whose parameters will be neural network
            function.
        neural_net_function:
            Specific architecture e.g. MultiLayerPerceptron that transforms
            input into parameters of the distribution.
        **kwargs:
            Furtuer parameters to be passed to the neural_net_function i.e.
            input tensor, layers, etc.
        """
        self.dist = None
        self.param_a = neural_net_function(**kwargs).get_output_layer();
        self.param_b = neural_net_function(**kwargs).get_output_layer();
        if distribution is tf.distributions.Normal:
            # Rectify standard deviation so that it is a smooth
            # positive function
            self.param_b = tf.nn.softmax(self.param_b + 1e-6)
            self.dist = tf.distributions.Normal(
                    loc=self.param_a, scale=self.param_b)
        elif distribution is LogitNormal:
            self.param_b = tf.nn.softmax(self.param_b + 1e-6)
            self.dist = LogitNormal(
                    loc=self.param_a, scale=self.param_b)

    def get_distribution(self):
        """Returns the reparameterized distributions.

        Returns:
        --------
        tensorflow.distributions.Distribution
        """
        return self.dist

    def log_prob(self, x):
        """Computes log probability of x under reparam distribution.

        Returns:
        --------
            tensorflow.Tensor that contains the log probability of input x.
        """
        return tf.reduce_sum(self.dist.log_prob(x), axis=-1)

    def sample(self, n_samples):
        """Samples from the reparameterized distribution.

        Parameters:
        -----------
        n_samples: int
            Number of samples.
        Returns:
        --------
            tensorflow.Tensor.
        """
        return self.dist.sample(n_samples)


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
            self.recon = self.decoder.mu

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

