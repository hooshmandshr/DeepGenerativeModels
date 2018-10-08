"""Provides the boiler plates and implementation for AEVB models.

The original AEVB paper:
https://arxiv.org/abs/1312.6114

Here, we provide abstract classes and implementation of certain models that are
used in our AEVB framework.
"""

import numpy as np
import tensorflow as tf

from distribution import LogitNormal


class Model(object):
    """Abstract class for implementing models in form p(x|y) or p(x)."""

    def __init__(self, out_dim, in_dim=0):
        """Sets up the computational graphs for the model.

        params:
        -------
        in_dim: int
            Dimensionality of the input parameters (if model is conditional).
            If 0, the model is not conditional meaning there is p(x).
        out_dim: int
            Dimensionality of the random variable that is goverend by models.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim

    def sample(self, n_samples, y=None):
        """Samples from the model conditional/or not."""
        pass

    def log_prob(self, x, y=None):
        """Computes log-prob of samples given the value of the distrib."""
        pass

    def entropy(self, y=None):
        """Computes the closed form entropy of the model if it exists."""
        pass


class ReparameterizedDistribution(Model):
    """Class that Implements reparameterized distributions p(x|y).

    p(x|y) is a known distribution q(x; f(y)) where the parameters of the
    distribution q() are governed by variable y through a transformation f().
    """

    def __init__(self, out_dim, in_dim, distribution, transform, **kwargs):
        """Initializes the model object and sets up necessary variables.

        params:
        -------
        out_dim: int
            Dimensionality of the output random variable x which is conditioned
            on the input variable.
        in_dim: int
            Dimensionality of the input (random) variable y.
        distribution: tensorflow.distributions.Distribution
            Known distribution that is reparameterized by a function f().
        transforma: Class that is extension of Transform
            Type of transformation f() to be applied to y in order to get
            the parameters of the know distributions.
        **kwargs:
            parameters to be passed to the Transform constructors.
        """    
        super(ReparameterizedDistribution, self).__init__(
                out_dim=out_dim, in_dim=in_dim)
        
        self.dist_class = distribution
        self.transform_class = transform
        # This list containts any concrete computation graph for this
        # reparametrized distribution given different inputs.
        self.dist_dict = {}
        self.trans_args = kwargs

    def get_distribution(self, y):
        """Get the tf.Distribution given y as an input for p(x|y).

        params:
        -------
        y: tf.Tensor
            Input that governs the conditional distribution p(x|y).

        returns:
        --------
        tf.Distribution for p(x|y) if y is None 
        """
        if not y.shape[-1].value == self.in_dim:
            raise ValueError(
                    "Input must of dimension {}".format(self.in_dim))
        if y in self.dist_dict:
            # Necessary tf.Distribution has already been created for y.
            return self.dist_dict[y]
        # Multivariate (Logit) normal with diagonal covariance.
        if self.dist_class is tf.distributions.Normal or\
                self.dist_class is LogitNormal:
            # Rectify standard deviation so that it is a smooth
            # positive function
            loc_ = self.transform_class(
                    in_dim=self.in_dim, out_dim=self.out_dim,
                    **self.trans_args).operator(y)
            scale_ = tf.nn.softmax(self.transform_class(
                    in_dim=self.in_dim, out_dim=self.out_dim,
                    **self.trans_args).operator(y))
            dist = self.dist_class(loc=loc_, scale=scale_)
        # Multivariate Poisson (independent variables).
        elif self.dist_class is tf.contrib.distributions.Poisson:
            rate_ = tf.nn.softmax(self.transform_class(
                    in_dim=self.in_dim, out_dim=self.out_dim,
                    **self.trans_args).operator(y))
            dist = self.dist_class(rate_)
        # Multivariate Normal With full covariance.
        elif self.dist_class is\
                tf.contrib.distributions.MultivariateNormalFullCovariance:
            loc_ = self.transform_class(
                    in_dim=self.in_dim, out_dim=self.out_dim,
                    **self.trans_args).operator(y)
            cov_ = self.transform_class(
                    in_dim=self.in_dim, out_dim=self.out_dim * self.out_dim,
                    **self.trans_args).operator(y)
            cov_ = tf.reshape(cov_, [-1, self.out_dim, self.out_dim])
            # Ensure that the covariance matrix is symmetric SDP.
            cov_ = tf.matrix_band_part(cov_, -1, 0)
            cov_ = tf.matmul(cov_, cov_, transpose_b=True)
            # Enforce postive definiteness.
            dist = self.dist_class(loc=loc_, covariance_matrix=cov_)

        # Store the created distribution for tensor y in the dictionary.
        self.dist_dict[y] = dist
        return dist

    def log_prob(self, x, y):
        """Computes log probability of x under reparam distribution.

        Returns:
        --------
            tensorflow.Tensor that contains the log probability of input x.
        """
        dist = self.get_distribution(y)
        if self.dist_class is tf.distributions.Normal or\
                self.dist_class is LogitNormal or\
                self.dist_class is tf.contrib.distributions.Poisson:
            return tf.reduce_sum(dist.log_prob(x), axis=-1)
        return dist.log_prob(x)

    def sample(self, y, n_samples):
        """Samples from the reparameterized distribution.

        Parameters:
        -----------
        n_samples: int
            Number of samples.
        Returns:
        --------
            tensorflow.Tensor.
        """
        dist = self.get_distribution(y)
        return dist.sample(n_samples)

    def entropy(self, y):
        """Samples from the reparameterized distribution.

        Parameters:
        -----------
        y: tf.Tensor
            Input variable that governs the distribution.
        Returns:
        --------
            tensorflow.Tensor.
        """
        dist = self.get_distribution(y)
        if self.dist_class is tf.distributions.Normal:
            return tf.reduce_sum(dist.entropy(), axis=-1)
        return dist.entropy()
