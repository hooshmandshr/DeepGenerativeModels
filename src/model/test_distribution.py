"""Testing correctness of the implemented distributions."""

import numpy as np
import scipy.stats
import tensorflow as tf

from distribution import *
from utils.test_block_matrix import dense_matrix

def test_block_diagonal_normal():
    """Tests the correctness of BlockTriDiagonalNormal distribution."""
    time, dim = 3, 2
    dtype = np.float32
    loc_ = np.random.rand(time * dim).astype(dtype)
    # diagonal part of the inverse covariance.
    inv_cov_d = np.random.rand(time, dim, dim)
    inv_cov_d += inv_cov_d.transpose([0, 2, 1])
    inv_cov_d /= 2.
    inv_cov_d += np.eye(dim)[None, :, :] 
    inv_cov_d = inv_cov_d.astype(dtype)
    # Off-diagonal block part of the inverse-covariance.
    inv_cov_o = np.random.rand(time - 1, dim, dim).astype(dtype)
    # Get the dense form of the matrix.
    dense_inv_cov = dense_matrix(inv_cov_d, inv_cov_o, tridiagonal=True)

    # Number of samples.
    n_ex = 10000
    with tf.Graph().as_default():
        btm = BlockTriDiagonalMatrix(
            diag_block=tf.constant(inv_cov_d),
            offdiag_block=tf.constant(inv_cov_o))
        dist = BlockTriDiagonalNormal(loc=loc_, inv_cov=btm)
        entropy_tensor = dist.entropy()
        samples_tensor = dist.sample(n_ex)
        with tf.Session() as sess:
            samples = sess.run(samples_tensor)
            entropy = sess.run(entropy_tensor)
    print "Checking correctness of entropy computation."
    cov = np.linalg.inv(dense_inv_cov)
    assert np.allclose(
            scipy.stats.multivariate_normal(loc_, cov).entropy(),
            entropy), "Correctness of entorpy."

    print "Checking the shape of samples"
    assert samples.shape == (n_ex, time * dim), "Shape of samples."

if __name__ == "__main__":
    test_block_diagonal_normal()
 
