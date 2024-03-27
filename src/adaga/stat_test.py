import gpflow
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

import scipy
import scipy.stats

import logging


class StatisticalTest(object):
    """
    Class that implements the statistical test, used for detecting window degeneracy.
    """

    def __init__(self, delta: float):
        """
        Args:
            model_current_expert (object): The model trained on the whole window.
            model_new_expert (object): The model trained on the overlap.
            delta (float): The delta value to use in the thresholds.

        """
        self.delta = delta

    def compute_covariance(
        self, model_0: gpflow.models.GPModel, model_1: gpflow.models.GPModel
    ) -> tuple:
        """Computes the covariance matrices for two given Gaussian process models.

        Args:
            model_0 (GaussianProcessModel): The first Gaussian process model.
            model_1 (GaussianProcessModel): The second Gaussian process model.

        Returns:
            A tuple containing four covariance matrices:
            - k_uf: Covariance matrix between inducing points and input data for model_0.
            - k_uu: Covariance matrix between inducing points for model_0.
            - h_uf: Covariance matrix between inducing points and input data for model_1.
            - h_uu: Covariance matrix between inducing points for model_1.
        """
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

    @tf.function
    def compute_alternative_cov_helper(
        self, model_data, kuu, kuf, huu, huf, sigma_sq, xi_sq
    ):
        alpha = 1 / sigma_sq + 1 / xi_sq

        d = (
            alpha * sigma_sq * kuu
            + alpha * tf.linalg.matmul(kuf, kuf, transpose_b=True)
            - (1 / sigma_sq) * tf.linalg.matmul(kuf, kuf, transpose_b=True)
        )
        d_inv_k_uf = tf.linalg.solve(d, kuf)
        a_inv = (
            1.0
            / alpha
            * (
                tf.eye(tf.shape(model_data)[0], dtype=tf.float64)
                + (1 / sigma_sq) * tf.linalg.matmul(kuf, d_inv_k_uf, transpose_a=True)
            )
        )

        c = (
            xi_sq * huu
            + tf.linalg.matmul(huf, huf, transpose_b=True)
            - (1 / xi_sq)
            * tf.linalg.matmul(huf, tf.linalg.matmul(a_inv, huf, transpose_b=True))
        )
        c_inv_h_mn_a_inv = tf.linalg.solve(c, tf.linalg.matmul(huf, a_inv))
        final_matrix = (1 / xi_sq) * tf.linalg.matmul(
            a_inv, tf.linalg.matmul(huf, c_inv_h_mn_a_inv, transpose_a=True)
        )
        res = a_inv + final_matrix

        return res

    def compute_alternative_cov(
        self, model_0: gpflow.models.GPModel, model_1: gpflow.models.GPModel
    ) -> tf.Tensor:
        """Computes the covariance matrix under the alternative hypothesis.

        Args:
            model_0 (object): The model trained on the window.
            model_1 (object): The model trained on the overlap.

        Returns:
            The covariance matrix under the alternative hypothesis.
        """
        k_uf, k_uu, h_uf, h_uu = self.compute_covariance(model_0, model_1)

        sigma_sq = model_0.likelihood.variance.numpy()
        xi_sq = model_1.likelihood.variance.numpy()

        result = self.compute_alternative_cov_helper(
            model_0.data[0],
            k_uu,
            k_uf,
            h_uu,
            h_uf,
            sigma_sq,
            xi_sq,
        )

        return result

    @tf.function
    def compute_expert_inv_covariance(
        self, model_data, model_inducing_points, kuu, kuf, sigma, variance
    ):
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAt = tf.linalg.matmul(A, A, transpose_b=True)
        B = AAt + tf.eye(
            tf.shape(model_inducing_points)[0], dtype=gpflow.default_float()
        )
        identity = (
            tf.eye(tf.shape(model_data)[0], dtype=gpflow.default_float()) / variance
        )
        Lb = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(Lb, A, lower=True) / sigma
        matrix = tf.linalg.matmul(tf.transpose(c), c)

        return tf.subtract(identity, matrix)

    def _compute_single_expert_inv_covariance(
        self, model: gpflow.models.GPModel
    ) -> tf.Tensor:
        """Computes the inverse covariance matrix for a given GP model.

        Args:
            model:
                The GP model whose inverse covariance matrix is to be computed.

        Returns:
            The inverse covariance matrix.
        """
        k_uf = gpflow.covariances.kufs.Kuf_kernel_inducingpoints(
            model.inducing_variable, model.kernel, model.data[0]
        )
        k_uu = gpflow.covariances.kuus.Kuu_kernel_inducingpoints(
            model.inducing_variable, model.kernel, jitter=gpflow.default_jitter()
        )
        variance = model.likelihood.variance.numpy()
        sigma = tf.sqrt(model.likelihood.variance.numpy())
        covariance_val = self.compute_expert_inv_covariance(
            model.data[0],
            model.inducing_variable.Z.numpy(),
            k_uu,
            k_uf,
            sigma,
            variance,
        )

        return covariance_val

    @tf.function
    def compute_expert_covariance(
        self, model_data: np.ndarray, kuu, kuf, variance: float
    ):
        L = tf.linalg.cholesky(kuu)
        c = tf.linalg.triangular_solve(L, kuf, lower=True)
        matrix = tf.linalg.matmul(tf.transpose(c), c)

        identity = variance * tf.eye(tf.shape(model_data)[0], dtype=tf.float64)

        # del L, c, kuu, kuf

        return tf.add(identity, matrix)

    def _compute_single_expert_covariance(
        self, model: gpflow.models.GPModel
    ) -> tf.Tensor:
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
        variance = model.likelihood.variance.numpy()
        covariance_val = self.compute_expert_covariance(
            model.data[0], k_uu, k_uf, variance
        )

        return covariance_val

    @tf.function
    def compute_ratio(
        self, inverse_cov_new_exp: tf.Tensor, vector: np.ndarray
    ) -> tf.Tensor:
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

    @tf.function
    def compute_thresholds_init(self, cov_null, cov_alt, inverse_cov_new_exp, alt):
        if alt:
            matrix = tf.linalg.matmul(cov_alt, inverse_cov_new_exp)
        else:
            matrix = tf.linalg.matmul(cov_null, inverse_cov_new_exp)

        trace = tf.linalg.trace(matrix)
        bound = -trace
        squared_l2_norm_eigvals = tf.linalg.trace(tf.linalg.matmul(matrix, matrix))
        return matrix, squared_l2_norm_eigvals, bound

    def compute_thresholds(
        self,
        cov_null: tf.Tensor,
        cov_alt: tf.Tensor,
        cov_new_exp: tf.Tensor,
        inverse_cov_new_exp: tf.Tensor,
        alt=False,
    ) -> float:
        """
        Computes the threshold for controlling type I and type II errors.

        Args:
            cov_null:
                The covariance matrix under the null hypothesis.
            cov_alt:
                The covariance matrix under the alternative hypothesis.
            cov_new_exp:
                The covariance matrix of the expert trained on the overlap.
            inverse_cov_new_exp:
                The inverse covariance matrix of the model trained on the overlap.
            alt
                Whether the threshold is for type II errors. Defaults to False.

        Returns:
            The threshold value.
        """

        # Geometric Interpretation: In geometric terms, the trace represents the sum of the eigenvalues of
        # the matrix. Eigenvalues convey information about stretching or compressing in different directions.
        # The trace captures the total stretching or compressing effect of the matrix.
        supp_seed = 0
        tolerance = None

        matrix, squared_l2_norm_eigvals, bound = self.compute_thresholds_init(
            cov_null, cov_alt, inverse_cov_new_exp, alt
        )

        while True:
            try:
                if tolerance is None:
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
                if tolerance is None:
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

    def test(
        self,
        model_current_expert: gpflow.models.GPModel,
        model_new_expert: gpflow.models.GPModel,
    ):
        """
        Method that tests if the current window is spoiled or not.

        Returns:
            bool: The result of the test.
        """
        self.model_current_expert = model_current_expert
        self.model_new_expert = model_new_expert

        product_covariance_matrix_null = self._compute_single_expert_covariance(
            self.model_current_expert
        )

        new_cov = self._compute_single_expert_covariance(self.model_new_expert)
        inv_new_cov = self._compute_single_expert_inv_covariance(self.model_new_expert)
        product_covariance_matrix_alt = self.compute_alternative_cov(
            self.model_current_expert, self.model_new_expert
        )

        threshold_null = self.compute_thresholds(
            product_covariance_matrix_null,
            product_covariance_matrix_alt,
            new_cov,
            inv_new_cov,
        )
        threshold_alt = self.compute_thresholds(
            product_covariance_matrix_null,
            product_covariance_matrix_alt,
            new_cov,
            inv_new_cov,
            alt=True,
        )
        ratio = self.compute_ratio(inv_new_cov, self.model_new_expert.data[1])

        # print("Threshold for type I errors:", threshold_null.numpy())
        # print("Threshold for type II errors:", threshold_alt.numpy())
        # print("Ratio:", ratio.numpy())

        result = threshold_alt >= threshold_null and ratio >= threshold_null

        return result
