import gpflow
from gpflow.logdensities import multivariate_normal

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import logging

from src.adaga.stat_test import StatisticalTest


class AdaptiveRegionalization(object):
    """
    Class that regionalizes the time domain in a streaming fashion.
    """

    def __init__(
        self,
        domain_data: np.ndarray,
        system_data: np.ndarray,
        delta: float,
        min_w_size: int,
        n_ind_pts: int,
        seed: int,
        batch_size: int,
        kern: str = "Matern12",
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
            self.slice_domain()

    def make_logistic_boundary(
        self, low_bound: float, high_bound: float
    ) -> tfp.Bijectors:
        """Make the boundaries for the transforms of kernels.

        Args:
            low_bound:
                The lower boundary of the scale.
            high_bound:
                The higher boundary of the scale.
        Returns:
            The object declaring the lower and higher boundaries.
        """
        low_bound = tf.cast(low_bound, dtype=tf.float64)
        high_bound = tf.cast(high_bound, dtype=tf.float64)
        affine = tfp.bijectors.Shift(low_bound)(
            tfp.bijectors.Scale(high_bound - low_bound)
        )
        sigmoid = tfp.bijectors.Sigmoid()

        # Chain Shift and Scale
        logistic = tfp.bijectors.Chain([affine, sigmoid])
        return logistic

    def slice_domain(self):
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

            k = gpflow.kernels.Periodic(base_kernel=base)

        elif self.kern == "Linear":
            k = gpflow.kernels.Linear()

        if not new and self.kern != "Periodic":
            variance_transform = self.make_logistic_boundary(
                low_bound=1e-8, high_bound=1.4
            )
            k.variance = gpflow.Parameter(
                value=1.0,
                transform=variance_transform,
            )

        # Add Mater52 Kernel to the selected kernel if we want to use the summation
        # Of kernels
        if use_kernel_sum:
            k = gpflow.kernels.Sum(
                [k, gpflow.kernels.Matern52(lengthscales=kernel_lengthscale)]
            )

        return k

    def make_observation_params(
        self,
        window: np.ndarray,
        new: bool,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """Standardize the observations for model input."""
        # Define x and y
        x = np.expand_dims(window[:, 0], axis=-1)
        y = np.expand_dims(window[:, 1], axis=-1)
        if not new:
            x_mean = np.mean(x)
            x_std = np.std(x)
            y_mean = np.mean(y)
            y_std = np.std(y)

        # Standardize
        x = (x - x_mean) / x_std
        y = (y - y_mean) / y_std
        return x, y, x_mean, x_std, y_mean, y_std

    def creat_expert(
        self,
        window: np.ndarray,
        new: bool,
        x_mean: float = None,
        x_std: float = None,
        y_mean: float = None,
        y_std: float = None,
    ) -> tuple[gpflow.models, float, float, float, float]:
        """
            This method creates an expert on the region of interest.

        Args:
            window (slice): The slice of data that supports the expert.
            new (bool): Whether the expert is trained on the overlap or not.
            x_mean (float): The mean to use in time standardization.
            x_std (float): The std dev to use in time standardization.
            y_mean (float): The mean to use in observations' standardization.
            y_std (float): The std dev to use in observations' standardization.

        Returns:
            tuple: A tuple containing the expert, together with the (potentially recomputed) time
                and observations' mean and standard deviation.
        """
        # make x and y arrays and standardize them.
        x, y, x_mean, x_std, y_mean, y_std = self.make_observation_params(
            window, new, x_mean, x_std, y_mean, y_std
        )

        # Choose some inducing points
        z_init = np.random.choice(
            x[:, 0], min(self.num_inducing_points, x.shape[0]), replace=False
        )
        z_init = np.expand_dims(z_init, axis=-1)

        # Determine our kernel instance
        kernel = self.select_kernel(new=new, use_kernel_sum=True)

        # Define the Main model based on the provided kernel and data
        expert = gpflow.models.SGPR(
            data=(x, y), kernel=kernel, inducing_variable=z_init
        )

        return expert, x_mean, x_std, y_mean, y_std

    def compute_covariance(
        self, model_1: gpflow.models = None, model_2: gpflow.models = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the covariances of the given models."""
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

    def optimize_model(self, model):
        """Optimize the given model for minimum loss."""
        expert_optimizer = gpflow.optimizers.Scipy()
        expert_optimizer.minimize(
            closure=model.training_loss,
            variables=model.trainable_variables,
            compile=True,
            tf_fun_args=dict(jit_compile=True),
        )

        return model

    def regionalize(self) -> None:
        """Apply ADAGA streaming GP regression"""
        start = self.x[0, 0]

        # Choose the end point based on 2X the minimum window size
        end = start + 2 * self.min_window_size
        close_current_window = False
        new_window = True

        # Main loop for CP calculations
        while True:
            # Clean unused variables in memory_this is mainly for memory optimization
            tf.keras.backend.clear_session()

            tf.random.set_seed(self.seed)

            # Define windows in tuples of (x,y)
            window = np.array(
                [
                    win
                    for win in np.column_stack((self.x, self.y))
                    if start <= win[0] < end
                ]
            )
            print("start, end:", start, end)

            if window.shape[0] <= 1:
                break
            
            # Define the comparison window starting point
            best_start_new_exp = end - self.min_window_size

            # Choose the tuples for the expert training
            window_current_expert = np.array(
                [win for win in window if start <= win[0] < end]
            )

            # Make the model for the current window
            model_current_expert, x_mean, x_std, y_mean, y_std = self.creat_expert(
                window_current_expert, False
            )

            # train the model for the minimum loss for this window
            model_current_expert = self.optimize_model(model=model_current_expert)

            if (
                min(end, self.x[-1, 0]) - start
                > self.min_window_size + 3
                > end - self.x[-1, 0]
            ):
                # Define the new window based on the minimum window size
                window_new_expert = np.array(
                    [win for win in window if best_start_new_exp <= win[0] < end]
                )

                # Define the new expert
                model_new_expert, _, _, _, _ = self.creat_expert(
                    window_new_expert, True, x_mean, x_std, y_mean, y_std
                )
                model_current_expert.data = model_new_expert.data

                # Optimize the new model
                model_new_expert = self.optimize_model(model_new_expert)

                # print("CURRENT MODEL", model_current_expert.as_pandas_table())

                # print("NEW MODEL", model_new_expert.as_pandas_table())

                # Make the statistical test object and apply the statistics test
                statistical_test = StatisticalTest(
                    model_current_expert, model_new_expert, self.delta
                )
                bad_current_window = statistical_test.test()

                # See if the current window is ruined
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
