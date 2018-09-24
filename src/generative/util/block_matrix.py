"""Tools for sparse representation of block(bi-tri)diagonal matrices.

Proveds classes block diagonal, block-bi-diagonal, and block-tri-diagonal
marices and respectively their inverce, solving a linear system (Ax=B, in
other words x=A-1 B), and cholesky factor decomposition.

1) Inverse of a BlockDiagonalMatrix is another BlockDiagonalMatrix.
2) Solving Ax=B for x where A is a BlockBiDiagnolMatrix is a tensorflow.Tensor
of shape of resulting matrix (i.e. A-1B).
3) Cholesky factor of a BlockTriDiagonalMatrix is a BlockBiDiagonalMatrix.
"""
import numpy as np
import tensorflow as tf


class BlockDiagonalMatrix(object):

    def __init__(self, diag_block):
        """Set up the blocks of the matrix (lower-bi-diagonal) by default.

        params:
        -------
        diag_block: tensorflow.Tensor of shape (T, D, D)
            Respectively the T Digaonal blocks of the matrix where each have
            D x D dimensions.
        """
        self.num_block = diag_block.shape[0].value
        self.block_dim = diag_block.shape[1].value
        self.diag_block = diag_block
       
    def matmul(self, b):
        """Matrix multiplication with another matrix."""
        pass

    def inverse(self):
        """Returns the inverse of the Block Diagonal Matrix."""
        # Diagonal blocks of the resulting matrix.
        res_diag = []
        for i in range(self.num_block):
            res_diag.append(tf.expand_dims(
                tf.linalg.inv(self.diag_block[i]), axis=0))
        return BlockDiagonalMatrix(
            diag_block=tf.concat(res_diag, axis=0))


class BlockBiDiagonalMatrix(BlockDiagonalMatrix):

    def __init__(self, diag_block, offdiag_block, lower=True):
        """Set up the blocks of the matrix (lower-bi-diagonal) by default.

        params:
        -------
        diag_block: tensorflow.Tensor of shape (T, D, D)
            Respectively the T Digaonal blocks of the matrix where each have
            D x D dimensions.
        offdiag_block: tensorflow.Tensor of shape (T-1, D, D)
            Respectively the T - 1 Digaonal blocks of the matrix where each
            have D x D dimensions.
        lower: bool
            True if the matrix is a lower-block-bi-diagonal matrix. Otherwise,
            upper-bi-diagonal-matrix.
        """
        super(BlockBiDiagonalMatrix, self).__init__(
            diag_block=diag_block)
        # Lower off diagonal blocks by default.
        self.offdiag_block = offdiag_block
        self.lower = lower

    def solve(self, b):
        """Returns x for which Ax=b where A is the matrix.

        params:
        -------
        b: tensorflow.Tensor of shape (N, self.num_block * self.block_dim).

        Returns:
        --------
        tensorflow.Tensor of shape (N, self.num_block * self.block_dim)
        which is the result of A^-1 * b.
        """
        # Diagonal blocks of the resulting matrix.
        def dot(M, x):
            return tf.reduce_sum(
                tf.expand_dims(M, axis=0) * tf.expand_dims(x, axis=1),
                axis=2)
        def b_slice(time):
            start_idx = time * self.block_dim
            return tf.slice(b, [0, start_idx], [-1, self.block_dim])

        if self.lower:
            x = []
            x.append(dot(tf.linalg.inv(self.diag_block[0]), b_slice(0)))
            for i in range(1, self.num_block):
                g = b_slice(i) - dot(self.offdiag_block[i-1], x[-1])
                x.append(dot(tf.linalg.inv(self.diag_block[i]), g))
        else:
            x = []
            x.append(dot(tf.linalg.inv(
                self.diag_block[-1]), b_slice(self.num_block - 1)))
            for i in range(0, self.num_block - 1)[::-1]:
                g = b_slice(i) - dot(self.offdiag_block[i], x[-1])
                x.append(dot(tf.linalg.inv(self.diag_block[i]), g))
            x.reverse()
        return tf.concat(x, axis=1)

    def transpose(self, in_place=False):
        """Transposes the matrix by transposing the blocks.

        params:
        -------
        in_place: bool
            If True, the transpose operation is done in place. Otherwise, a new
            BlockBiDiagonalMatrix is returned.

        returns:
        --------
        None or BlockBiDiagonalMatrix.
        """
        if in_place:
            self.diag_block = tf.transpose(self.diag_block, perm=[0, 2, 1])
            self.offdiag_block = tf.transpose(
                    self.offdiag_block, perm=[0, 2, 1])
            self.lower = not self.lower
        else:
            return BlockBiDiagonalMatrix(
                    diag_block=tf.transpose(self.diag_block, perm=[0, 2, 1]),
                    offdiag_block=tf.transpose(
                        self.offdiag_block, perm=[0, 2, 1]),
                    lower=not(self.lower))

class BlockTriDiagonalMatrix(BlockBiDiagonalMatrix):

    def __init__(self, diag_block, offdiag_block):
        """Set up the blocks of the matrix (lower-bi-diagonal) by default.

        params:
        -------
        diag_block: tensorflow.Tensor of shape (T, D, D)
            Respectively the T Digaonal blocks of the matrix where each have
            D x D dimensions.
        offdiag_block: tensorflow.Tensor of shape (T-1, D, D)
            Respectively the T - 1 Digaonal blocks of the matrix where each
            have D x D dimensions. 
        """

        super(BlockTriDiagonalMatrix, self).__init__(
            diag_block=diag_block, offdiag_block=offdiag_block)

    def cholesky(self):
        """Coputes the cholesky factor (lower triangular) of the matrix.

        returns:
        --------
        BlockBiDiagonalMatrix which represents the lower tirangular cholesky
        factor of block-tri-diagonal matrix that the objects represents.
        """
        result_diag = []
        result_offdiag = []
        result_diag.append(self.diag_block[0])
        for i in range(self.num_block - 1):
            result_diag[i] = tf.cholesky(result_diag[i])
            result_offdiag.append(tf.matmul(
                self.offdiag_block[i],
                tf.linalg.inv(result_diag[i]), transpose_b=True))
            result_diag.append(self.diag_block[i + 1] - tf.matmul(
                result_offdiag[i], result_offdiag[i], transpose_b=True))
        result_diag[-1] = tf.cholesky(result_diag[-1])
        # Concatenating the tensors into higher dimensional tensor.
        def expand(tensor_list):
            return tf.concat(
                [tf.expand_dims(block, axis=0) for block in tensor_list],
                axis=0)
        result_diag = expand(result_diag)
        result_offdiag = expand(result_offdiag)
        return BlockBiDiagonalMatrix(
            diag_block=result_diag, offdiag_block=result_offdiag)

