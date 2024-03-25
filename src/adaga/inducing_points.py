import gpflow
from gpflow.logdensities import multivariate_normal

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import logging

import gc

from src.adaga.stat_test import StatisticalTest


class AdaptiveRegionalization(object):
    """
    Class that regionalizes the time domain in a streaming fashion.
    """

    def __init__(
        self,
        domain_data,
        system_data,
        delta,
        min_w_size,
        n_ind_pts,
        seed,
        batch_size,
        kern="RBF",
        domain_test=None,
        system_test=None,
        input_horizon=None,
    ):
        """
        Constructor
        :param domain_data: [n x 1] array of timesteps;
        :param system_data: [n x 1] array of observations;
        :param delta: the delta hyperparameter to be used in the thresholds;
        :param min_w_size: the minimum window size allowed;
        :param n_ind_pts: the number of inducing points to use;
        :param seed: the seed (fixed for reproducibility);
        :param n_batches: the number of batches in which the overll trajectory is partitioned;
        :param kern: the kernel to be used.
        """
        self.x = domain_data
        self.y = system_data
        self.n_states, self.n_points = system_data.shape
        self.delta = delta
        self.min_window_size = min_w_size
        self.num_inducing_points = n_ind_pts
        self.batch_time_jump = batch_size
        self.seed = seed
        self.closed_windows = []
        self.kern = kern
        self.domain_test = domain_test
        self.y_test = system_test
        # This param is needed to decouple the minimum window size from a reduced time horizon
        # (w.r.t. the final value of the trajectory).
        self.input_horizon = input_horizon
        if self.input_horizon is not None:
            self._slice_domain_function()

    def make_logistic_boundary(self, low_bound, high_bound):
        low_bound = tf.cast(low_bound, dtype=tf.float64)
        high_bound = tf.cast(high_bound, dtype=tf.float64)
        affine = tfp.bijectors.Shift(low_bound)(
            tfp.bijectors.Scale(high_bound - low_bound)
        )  # Chain Shift and Scale
        sigmoid = tfp.bijectors.Sigmoid()
        logistic = tfp.bijectors.Chain([affine, sigmoid])
        return logistic

    def _slice_domain_function(self):
        sliced_x_y = np.array(
            [e for e in np.column_stack((self.x, self.y)) if e[0] <= self.input_horizon]
        )
        self.x = np.expand_dims(sliced_x_y[:, 0], axis=-1)
        self.y = np.expand_dims(sliced_x_y[:, 1], axis=-1)

    def select_kernel(self, new: bool, use_kernel_sum: bool = True):
        kernel_lengthscale = gpflow.Parameter(
            value=1,
            transform=self.make_logistic_boundary(1e-3, 100),
            name="Kernel_lengthscale",
        )

        if self.kern == "RBF":
            k = gpflow.kernels.RBF(lengthscales=kernel_lengthscale)
        elif self.kern == "Matern52":
            k = gpflow.kernels.Matern52(lengthscales=kernel_lengthscale)
        elif self.kern == "Matern32":
            k = gpflow.kernels.Matern32(lengthscales=kernel_lengthscale)
        elif self.kern == "Matern12":
            k = gpflow.kernels.Matern12(lengthscales=kernel_lengthscale)
        elif self.kern == "RQ":
            k = gpflow.kernels.RationalQuadratic(lengthscales=kernel_lengthscale)
        elif self.kern == "Periodic":
            base = gpflow.kernels.RBF(
                lengthscales=kernel_lengthscale,
            )

            if not new:
                variance_transform = self.make_logistic_boundary(
                    low_bound=1e-8, high_bound=1.4
                )
                base.variance = gpflow.Parameter(
                    value=1.0, transform=variance_transform, name="Base_Variance"
                )
            # else:
            #     base.variance = 1.0

            k = gpflow.kernels.Periodic(base_kernel=base)

        elif self.kern == "Linear":
            k = gpflow.kernels.Linear()

        # k.variance = 1.0

        if not new and self.kern != "Periodic":
            variance_transform = self.make_logistic_boundary(
                low_bound=1e-8, high_bound=1.4
            )
            k.variance = gpflow.Parameter(
                value=1.0,
                transform=variance_transform,
            )

        # Add Mater52 Kernel to the selected kernel if we want to have an addition
        # To our kernel
        if use_kernel_sum:
            k = gpflow.kernels.Sum(
                [k, gpflow.kernels.Matern52(lengthscales=kernel_lengthscale)]
            )

        return k

    def _create_expert(
        self, window, new: bool, x_mean=None, x_std=None, y_mean=None, y_std=None
    ):
        """
        This method creates the expert on the region of interest (full window or overlap);
        :param window: the slice of data that supports the expert;
        :param new: whether the expert is trained on the overlap or not;
        :param x_mean: the mean to use in time standardization;
        :param x_std: the std dev to use in time standardization;
        :param y_mean: the mean to use in observations' standardization;
        :param y_std: the std dev to use in observations' standardization;
        :return: the expert, together with the (potentially recomputed) time and observations' mean, atd dev.
        """
        x = np.expand_dims(window[:, 0], axis=-1)
        y = np.expand_dims(window[:, 1], axis=-1)
        if not new:
            x_mean = np.mean(x)
            x_std = np.std(x)
            y_mean = np.mean(y)
            y_std = np.std(y)
        x -= x_mean
        x /= x_std
        y -= y_mean
        y /= y_std
        z_init = np.random.choice(
            x[:, 0], min(self.num_inducing_points, x.shape[0]), replace=False
        )
        z_init = np.expand_dims(z_init, axis=-1)

        kernel = self.select_kernel(new=new, use_kernel_sum=True)

        # variance_trainable = True
        # kernel_variance = gpflow.Parameter(value=1.0, transform=variance_transform, name="Kernel_variance")
        # else:
        #     variance_transform = None
        #     variance_trainable = False
        # kernel_variance = None

        # k.variance = gpflow.Parameter(
        #     value=1.0,
        #     transform=variance_transform,
        # name="Kernel_variance",
        # trainable=variance_trainable
        # )

        expert = gpflow.models.SGPR(
            data=(x, y), kernel=kernel, inducing_variable=z_init
        )

        return expert, x_mean, x_std, y_mean, y_std

    def _build_likelihood(self, model: gpflow.models.SGPR):
        """
        This method builds the likelihood of the given model.
        """

        K_uf = gpflow.covariances.kufs.Kuf_kernel_inducingpoints(
            model.inducing_variable, model.kernel, model.data[0]
        )
        K_uu = gpflow.covariances.kuus.Kuu_kernel_inducingpoints(
            model.inducing_variable, model.kernel, jitter=gpflow.default_jitter()
        )
        K_uu_inv = np.linalg.inv(K_uu)
        K = (
            np.matmul(np.matmul(np.transpose(K_uf), K_uu_inv), K_uf)
            + np.identity(model.data[0].shape[0], dtype=gpflow.default_float())
            * model.likelihood.variance.numpy()
        )
        L = np.linalg.cholesky(K)
        m = model.mean_function(model.data[0])
        y_tensor = tf.constant(model.data[1])
        logpdf = multivariate_normal(
            y_tensor, m, L
        )  # (R,) log-likelihoods for each independent dimension of Y

        return np.sum(logpdf)

    def compute_covariance(self, model_1=None, model_2=None):
        K_1, K_2 = None, None
        cov_kuf = gpflow.covariances.kufs.Kuf_kernel_inducingpoints
        cov_kuu = gpflow.covariances.kuus.Kuu_kernel_inducingpoints
        if model_1:
            K_uf_1 = cov_kuf(model_1.inducing_variable, model_1.kernel, model_1.data[0])
            K_uu_1 = cov_kuu(
                model_1.inducing_variable,
                model_1.kernel,
                jitter=gpflow.default_jitter(),
            )
            K_uu_inv_1 = np.linalg.inv(K_uu_1)
            K_1 = (
                np.matmul(np.matmul(np.transpose(K_uf_1), K_uu_inv_1), K_uf_1)
                + np.identity(model_1.data[0].shape[0], dtype=gpflow.default_float())
                * model_1.likelihood.variance.numpy()
            )
        if model_2:
            K_uf_2 = cov_kuf(model_2.inducing_variable, model_2.kernel, model_2.data[0])
            K_uu_2 = cov_kuu(
                model_2.inducing_variable,
                model_2.kernel,
                jitter=gpflow.default_jitter(),
            )
            K_uu_inv_2 = np.linalg.inv(K_uu_2)
            K_2 = (
                np.matmul(np.matmul(np.transpose(K_uf_2), K_uu_inv_2), K_uf_2)
                + np.identity(model_2.data[0].shape[0], dtype=gpflow.default_float())
                * model_2.likelihood.variance.numpy()
            )

        return K_1, K_2

    def _build_norm_const(
        self, model_1: gpflow.models.SGPR, model_2: gpflow.models.SGPR
    ):
        """
        This method builds the normalization constant of the product of two Gaussian pdfs.
        """
        K_1, K_2 = self.compute_covariance(model_1=model_1, model_2=model_2)

        L = np.linalg.cholesky(K_1 + K_2)
        m_1 = model_1.mean_function(model_1.datat[0])
        m_2 = model_2.mean_function(model_2.data[0])
        logpdf = multivariate_normal(
            m_1, m_2, L
        )  # (R,) log-likelihoods for each independent dimension of Y

        return np.sum(logpdf)

    def _build_norm_const_new(self, model_1: gpflow.models.SGPR):
        """
        This method builds the normalization constant of a given model.
        """
        K_1 = self.compute_covariance(model_1=model_1, model_2=model_1)

        _, c = np.linalg.slogdet(a=2 * np.pi * K_1)
        return 0.5 * c

    def test(self):
        final_pred = np.empty((0, 1))
        final_gt = np.empty((0, 1))
        final_time = np.empty((0, 1))
        for region in self.closed_windows:
            window_test = np.array(
                [
                    e
                    for e in np.column_stack((self.domain_test, self.y_test))
                    if region["window_start"] <= e[0] < region["window_end"]
                ]
            )

            model_test, x_mean_test, x_std_test, y_mean_test, y_std_test = (
                self._create_expert(window_test, False)
            )
            opt_test = gpflow.train.ScipyOptimizer()
            opt_test.minimize(model_test)

            x_pred_test = np.expand_dims(window_test[:, 0], axis=-1)
            pred, _ = model_test.predict_f(x_pred_test)
            pred = pred * y_std_test + y_mean_test
            y_gt = np.expand_dims(window_test[:, 1], axis=-1) * y_std_test + y_mean_test

            final_pred = np.concatenate((final_pred, pred), axis=0)
            final_gt = np.concatenate((final_gt, y_gt), axis=0)
            final_time = np.concatenate(
                (final_time, x_pred_test * x_std_test + x_mean_test), axis=0
            )

        self.rmse = [
            self.x.shape[0],
            np.sqrt(np.sum((final_pred - final_gt) ** 2) / final_gt.shape[0]),
        ]

    def optimize_model(self, model):
        expert_optimizer = gpflow.optimizers.Scipy()
        expert_optimizer.minimize(
            closure=model.training_loss,
            variables=model.trainable_variables,
            compile=True,
            tf_fun_args=dict(jit_compile=True),
        )

        return model

    def regionalize(self) -> None:
        """
        This method applies ADAGA streaming GP regression.
        """
        start = self.x[0, 0]
        end = start + 2 * self.min_window_size  # + self.batch_time_jump
        close_current_window = False
        new_window = True
        while True:
            tf.keras.backend.clear_session()
            gc.collect()

            tf.random.set_seed(self.seed)
            window = np.array(
                [e for e in np.column_stack((self.x, self.y)) if start <= e[0] < end]
            )
            print("start, end:", start, end)
            # print("WINDOW SHAPE", window.shape)

            if window.shape[0] <= 1:
                break

            best_start_new_exp = end - self.min_window_size

            window_current_expert = np.array([e for e in window if start <= e[0] < end])
            model_current_expert, x_mean, x_std, y_mean, y_std = self._create_expert(
                window_current_expert, False
            )

            model_current_expert = self.optimize_model(model=model_current_expert)

            if (
                min(end, self.x[-1, 0]) - start
                > self.min_window_size + 3
                > end - self.x[-1, 0]
            ):
                window_new_expert = np.array(
                    [e for e in window if best_start_new_exp <= e[0] < end]
                )
                model_new_expert, _, _, _, _ = self._create_expert(
                    window_new_expert, True, x_mean, x_std, y_mean, y_std
                )
                model_current_expert.data = model_new_expert.data

                model_new_expert = self.optimize_model(model_new_expert)

                # print("CURRENT MODEL", model_current_expert.as_pandas_table())

                # print("NEW MODEL", model_new_expert.as_pandas_table())
                statistical_test = StatisticalTest(
                    model_current_expert, model_new_expert, self.delta
                )
                bad_current_window = statistical_test.test()

                if bad_current_window:
                    close_current_window = True

            if not close_current_window or new_window:
                new_window = False

            if end > self.x[-1] or close_current_window:
                new_window = True
                end_test = (
                    self.x[-1, 0] if end > self.x[-1] else end - self.min_window_size
                )

                self.closed_windows.append(
                    {"window_start": start, "window_end": end_test}
                )

                start = end - self.min_window_size

                if end > self.x[-1]:
                    break
                end = start + 2 * self.min_window_size

            if (
                not close_current_window
                or end - start < self.min_window_size
                or new_window
            ):
                end += self.batch_time_jump

            close_current_window = False
        print(
            "PARTITIONING CREATED:",
            [(e["window_start"], e["window_end"]) for e in self.closed_windows],
        )
