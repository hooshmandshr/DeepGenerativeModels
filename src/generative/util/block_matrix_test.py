
import numpy as np
import tensorflow as tf

import block_matrix as bm


def dense_matrix(diag, offdiag=None, tridiagonal=False):
    """Returns a dense version of a block diagonal matrix given block representation.

    params:
    -------
    diag: np.ndarray of shape (T, D, D)
        Represents T respective D x D diagonal blocks of a TD x TD matrix.
    offdiag: np.ndarray of shape (T-1, D, D)
        Represents T - 1 respective D x D off diagonal blocks of a
        TD x TD matrix. If None, the matrix is block diagonal.
    tridiagonal: bool
        If True, the matrix will be block-tri-diagonal. Otherwise,
        block-bi-diagonal.
    returns:
    --------
    np.ndarray of shape (T * D, T * D).
    """
    n_block = diag.shape[0]
    block_dim = diag.shape[1]
    tot_dim = n_block * block_dim
    matrix = np.zeros([tot_dim, tot_dim])
    for i in range(n_block):
        row = i * block_dim
        dim = block_dim
        matrix[row:(row + dim), row:(row + dim)] = diag[i]
        if i > 0 and offdiag is not None:
            if tridiagonal:
                matrix[(row - dim):row, row:(row + dim)] = offdiag[i - 1].T
            matrix[row:(row + dim), (row - dim):row] = offdiag[i - 1]
    return matrix

def test_diag_inverse():
    """Tests whether inverse computation of block matrix is correct."""
    time = 3
    dim = 4
    diag = np.random.rand(time, dim, dim)
    # Setting up tensors for the corresponding blocks.
    with tf.Graph().as_default():
        bd_tensor = bm.BlockDiagonalMatrix(tf.constant(diag))
        inv = bd_tensor.inverse().diag_block
        with tf.Session() as sess:
            tf_res = sess.run(inv)
    dense_inverse = np.linalg.inv(dense_matrix(diag))
    dense_result = dense_matrix(tf_res)
    print "Testing inverse computation."
    assert np.allclose(dense_inverse, dense_result), 'Inverse computatoin'

def test_cholesky():
    """Tests whether inverse computation of block matrix is correct."""
    time = 3
    dim = 5
    # Enforce symmetry in the diagonal blocks.
    # Also, add a multiple of np.eye to guarantee decomposition.
    diag = np.random.rand(time, dim, dim)
    diag = (diag + diag) / 2.
    diag += np.array([np.eye(dim) * dim * time for i in range(time)])
    offdiag = np.random.rand(time - 1, dim, dim)

    b = np.random.rand(3, time * dim)
    # Setting up tensors for the corresponding blocks.
    with tf.Graph().as_default():
        bd_tensor = bm.BlockTriDiagonalMatrix(
            diag_block=tf.constant(diag), offdiag_block=tf.constant(offdiag))
        chl_matrix = bd_tensor.cholesky()
        # Cholesky factor blocks.
        chl_diag_tensor = chl_matrix.diag_block
        chl_offdiag_tensor = chl_matrix.offdiag_block
        # Inverse Cholesky solve.
        solve_tensor = chl_matrix.solve(tf.constant(b))
        chl_matrix.transpose(in_place=True)
        solve_transpose_tensor = chl_matrix.solve(tf.constant(b))
        with tf.Session() as sess:
            chl_res_diag, chl_res_offdiag, solve_res, solve_t_res = sess.run(
                [chl_diag_tensor, chl_offdiag_tensor, solve_tensor,
                    solve_transpose_tensor])
    dense_cholesky = np.linalg.cholesky(dense_matrix(diag, offdiag))
    dense_result = dense_matrix(chl_res_diag, chl_res_offdiag)
    print "Testing cholesky factor computation."
    assert np.allclose(dense_cholesky, dense_result), 'Choesky computatoin'

    # Solve choleskey system given b. In other words, compute the inverse
    # cholesky multplied by b.
    dense_solve_cholesky = np.matmul(np.linalg.inv(dense_cholesky), b.T)
    print "Testing cholesky inverse."
    assert np.allclose(dense_solve_cholesky.T, solve_res), 'Cholesky inverse'


    # Solve choleskey system given b if the matrix is transposed. In otherwords
    # we want the result of A^{-T}b.
    dense_solve_cholesky = np.matmul(np.linalg.inv(dense_cholesky).T, b.T)
    print "Testing cholesky transpose inverse."
    assert np.allclose(dense_solve_cholesky.T, solve_t_res), 'Cholesky transpose inverse'



if __name__ == "__main__":
    test_diag_inverse()
    test_cholesky()
