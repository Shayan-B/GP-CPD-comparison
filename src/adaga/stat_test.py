import gpflow
from gpflow.logdensities import multivariate_normal

import numpy as np

import tensorflow as tf

import scipy
import scipy.stats

import logging


class StatisticalTest(object):
    """
    Class that implements the statistical test, used for detecting window degeneracy.
    """

    def __init__(self, model_current_expert, model_new_expert, delta):
        """
        Constructor.
        :param model_current_expert: the model trained on the whole window;
        :param model_new_expert: the model trained on the overlap;
        :param delta: the delta value to use in the thresholds.
        """
        self.model_current_expert = model_current_expert
        self.model_new_expert = model_new_expert
        self.delta = delta

    def compute_covariance(self, model_0, model_1):
        cov_kuf = gpflow.covariances.kufs.Kuf_kernel_inducingpoints
        cov_kuu = gpflow.covariances.kuus.Kuu_kernel_inducingpoints

        k_uf = cov_kuf(model_0.inducing_variable, model_0.kernel, model_0.data[0])
        k_uu = cov_kuu(
            model_0.inducing_variable, model_0.kernel, jitter=gpflow.default_jitter()
        )
        h_uf = cov_kuf(model_1.inducing_variable, model_1.kernel, model_1.data[0])
        h_uu = cov_kuu(
            model_1.inducing_variable, model_1.kernel, jitter=gpflow.default_jitter()
        )

        return k_uf, k_uu, h_uf, h_uu

    def _compute_cov_alt(self, model_0, model_1) -> tf.Tensor:
        """
        Compute the covariance matrix under the alternative hypothesis;
        :param model_0: the model trained on the window;
        :param model_1: the model trained on the overlap;
        :return: the covariance matrix under the alternative hypothesis.
        """
        k_uf, k_uu, h_uf, h_uu = self.compute_covariance(model_0, model_1)

        sigma_sq = model_0.likelihood.variance.numpy()
        xi_sq = model_1.likelihood.variance.numpy()
        alpha = 1 / sigma_sq + 1 / xi_sq

        d = (
            alpha * sigma_sq * k_uu
            + alpha * tf.linalg.matmul(k_uf, k_uf, transpose_b=True)
            - (1 / sigma_sq) * tf.linalg.matmul(k_uf, k_uf, transpose_b=True)
        )
        d_inv_k_uf = tf.linalg.solve(d, k_uf)
        a_inv = (
            1.0
            / alpha
            * (
                tf.eye(tf.shape(model_0.data[0])[0], dtype=tf.float64)
                + (1 / sigma_sq) * tf.linalg.matmul(k_uf, d_inv_k_uf, transpose_a=True)
            )
        )

        c = (
            xi_sq * h_uu
            + tf.linalg.matmul(h_uf, h_uf, transpose_b=True)
            - (1 / xi_sq)
            * tf.linalg.matmul(h_uf, tf.linalg.matmul(a_inv, h_uf, transpose_b=True))
        )
        c_inv_h_mn_a_inv = tf.linalg.solve(c, tf.linalg.matmul(h_uf, a_inv))
        final_matrix = (1 / xi_sq) * tf.linalg.matmul(
            a_inv, tf.linalg.matmul(h_uf, c_inv_h_mn_a_inv, transpose_a=True)
        )
        result = a_inv + final_matrix

        return result

    def _compute_single_expert_inv_covariance(self, model) -> tf.Tensor:
        """
        Computes the inverse covariance matrix for a given GP model.
        :param model: the model whose inverse covariance matrix is to be computed;
        :return: the inverse covariance matrix.
        """
        k_uf = gpflow.covariances.kufs.Kuf_kernel_inducingpoints(
            model.inducing_variable, model.kernel, model.data[0]
        )
        k_uu = gpflow.covariances.kuus.Kuu_kernel_inducingpoints(
            model.inducing_variable, model.kernel, jitter=gpflow.default_jitter()
        )
        variance = model.likelihood.variance.numpy()
        sigma = tf.sqrt(model.likelihood.variance.numpy())
        L = tf.linalg.cholesky(k_uu)
        A = tf.linalg.triangular_solve(L, k_uf, lower=True) / sigma
        AAt = tf.linalg.matmul(A, A, transpose_b=True)
        B = AAt + tf.eye(
            tf.shape(model.inducing_variable.Z.numpy())[0], dtype=gpflow.default_float()
        )
        identity = (
            tf.eye(tf.shape(model.data[0])[0], dtype=gpflow.default_float()) / variance
        )
        Lb = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(Lb, A, lower=True) / sigma
        matrix = tf.linalg.matmul(tf.transpose(c), c)

        return tf.subtract(identity, matrix)

    def _compute_single_expert_covariance(self, model) -> tf.Tensor:
        """
        Computes the covariance matrix for a given GP model.
        :param model: the model whose covariance matrix is to be computed;
        :return: the covariance matrix.
        """

        k_uf = gpflow.covariances.kufs.Kuf_kernel_inducingpoints(
            model.inducing_variable, model.kernel, model.data[0]
        )
        k_uu = gpflow.covariances.kuus.Kuu_kernel_inducingpoints(
            model.inducing_variable, model.kernel, jitter=gpflow.default_jitter()
        )
        L = tf.linalg.cholesky(k_uu)
        c = tf.linalg.triangular_solve(L, k_uf, lower=True)
        matrix = tf.linalg.matmul(tf.transpose(c), c)

        variance = model.likelihood.variance.numpy()
        identity = variance * tf.eye(tf.shape(model.data[0])[0], dtype=tf.float64)

        return tf.add(identity, matrix)

    def _compute_ratio(self, inverse_cov_new_exp, vector) -> tf.Tensor:
        """
        Computes the test statistic.
        :param inverse_cov_new_exp: the inverse covariance matrix of the expert trained on the overlap;
        :param vector: the vector containing the observations;
        :return: the statistic value.
        """
        likelihood = -tf.linalg.matmul(
            tf.linalg.matmul(tf.transpose(vector), inverse_cov_new_exp), vector
        )
        return likelihood

    def _compute_thresholds(
        self, cov_null, cov_alt, cov_new_exp, inverse_cov_new_exp, alt=False
    ):
        """
        Computes the threshold for controlling type I and type II errors.
        :param cov_null: the covariance matrix under the null hypothesis;
        :param cov_alt: the covariance matrix under the alternative hypothesis;
        :param cov_new_exp: the covariance matrix of the expert trained on the overlap;
        :param inverse_cov_new_exp: the inverse covariance matrix of the model trained on the overlap.
        :param alt: wheteher the threshold is for type II errors, or not;
        :return: the threshold value.
        """
        n = tf.shape(cov_new_exp)[0]

        if alt:
            matrix = tf.linalg.matmul(cov_alt, inverse_cov_new_exp)
        else:
            matrix = tf.linalg.matmul(cov_null, inverse_cov_new_exp)

        # Geometric Interpretation: In geometric terms, the trace represents the sum of the eigenvalues of
        # the matrix. Eigenvalues convey information about stretching or compressing in different directions.
        # The trace captures the total stretching or compressing effect of the matrix.
        trace = tf.linalg.trace(matrix)
        bound = -trace
        squared_l2_norm_eigvals = tf.linalg.trace(tf.linalg.matmul(matrix, matrix))
        supp_seed = 0
        tolerance = None

        while True:
            try:
                if tolerance == None:
                    approx_infty_norm = scipy.sparse.linalg.eigs(
                        matrix.numpy(), k=1, which="LM", return_eigenvectors=False
                    )
                else:
                    approx_infty_norm = scipy.sparse.linalg.eigs(
                        matrix.numpy(),
                        k=1,
                        which="LM",
                        return_eigenvectors=False,
                        tol=tolerance,
                    )

                break
            except ValueError as e:
                print(e)
                if tolerance == None:
                    tolerance = 1e-15
                else:
                    tolerance *= 10
                if supp_seed > 15:
                    print("!FAILED CONVERGENCE OF LARG. EIGVAL! Too many times.")
                    exit(1)
                else:
                    print(
                        f"!FAILED CONVERGENCE OF LARG. EIGVAL! Reinitializing seed and increasing tolerance to {tolerance}."
                    )
                    np.random.seed(supp_seed)
                    supp_seed += 1

        nu_squared = tf.cast(4.0 * squared_l2_norm_eigvals, tf.float64)
        alpha = tf.cast(4.0 * approx_infty_norm, tf.float64)
        delta = self.delta
        log_term = 2.0 * tf.cast(tf.math.log(1.0 / delta), tf.float64)

        v = tf.maximum(tf.math.sqrt(log_term * nu_squared), log_term * alpha)
        const = v
        if alt:
            bound -= const
        else:
            bound += const

        return bound

    def test(self):
        """
        Method that tests if the current window is spoiled or not.
        :return: the result of the test (boolean).
        """
        product_covariance_matrix_null = self._compute_single_expert_covariance(
            self.model_current_expert
        )

        new_cov = self._compute_single_expert_covariance(self.model_new_expert)
        inv_new_cov = self._compute_single_expert_inv_covariance(self.model_new_expert)
        product_covariance_matrix_alt = self._compute_cov_alt(
            self.model_current_expert, self.model_new_expert
        )
        # print("Product:", product_covariance_matrix_null)

        threshold_null = self._compute_thresholds(
            product_covariance_matrix_null,
            product_covariance_matrix_alt,
            new_cov,
            inv_new_cov,
        )
        threshold_alt = self._compute_thresholds(
            product_covariance_matrix_null,
            product_covariance_matrix_alt,
            new_cov,
            inv_new_cov,
            alt=True,
        )
        ratio = self._compute_ratio(inv_new_cov, self.model_new_expert.data[1])

        print("Threshold for type I errors:", threshold_null.numpy())
        print("Threshold for type II errors:", threshold_alt.numpy())
        print("Ratio:", ratio.numpy())

        if threshold_alt >= threshold_null and ratio >= threshold_null:
            result = True
        else:
            result = False
        # print("Result of the test:", result)
        return result