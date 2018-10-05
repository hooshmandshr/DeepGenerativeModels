"""Provides extra dsitributions beside available tensorflow distributions."""


import numpy as np
import tensorflow as tf

from utils.block_matrix import *


class LogitNormal(tf.distributions.Normal):
    """Class that extends tf.distributions.Normal to a logit-normal distribution."""

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


class BlockTriDiagonalNormal(tf.distributions.Distribution):
    """Class for a multi-variate normal with block-tri-diagonal covariance."""

    def __init__(self, loc, inv_cov): 
        """Sets up the variables for the block-tri-diagonal covariance normal.

        params:
        -------
        loc: tf.Tensor:
            Mean of the multi-variate normal distribution.
        inv_cov: BlockTriDiagonalMatrix
            Inverse of the covariance of the multi-variate normal distribution.
            The shape must be compatible with loc.
        """
        # Noise variable that will be re-parameterized to produce
        # samples from the desired distribution with log density.
        if not isinstance(inv_cov, BlockTriDiagonalMatrix):
            raise ValueError(
                    "inv_cov should be of type BlockTriDiagonalMatrix.")

        self.base_dist = tf.distributions.Normal(
                loc=tf.zeros(loc.shape), scale=tf.ones(loc.shape))
        self.loc = loc
        self.inv_cov = inv_cov
        self.chol_factor = self.inv_cov.cholesky()
        self.chol_factor_transpose = self.chol_factor.transpose(in_place=False)

    def sample(self, n_samples):
        """Samples n_samples times from the mutli-variate normal."""
        samples = self.base_dist.sample(n_samples)
        return self.loc + self.chol_factor_transpose.solve(samples)

    def entropy(self):
        """Closed form entropy using det of cholesky factor of inv-cov."""
        tot_dim = self.chol_factor.num_block * self.chol_factor.block_dim
        const = (tf.log(2. * np.pi) + 1.) * tot_dim / 2.
        return - tf.reduce_sum(tf.reduce_sum(
                tf.log(self.chol_factor.get_diag_part()), axis=-1), axis=-1) + const

