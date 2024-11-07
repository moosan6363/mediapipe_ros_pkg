import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter as KalmanFilterPy


class KalmanFilter:
    def __init__(
        self,
        dim_x: int,  # state_dimension
        dim_z: int,  # measurement_dimension
        h,  # measurement_matrix
        x_0,  # initial_state_matrix
        p,  # initial_state_covariance_matrix
        r,  # measurement_noise_covariance_matrix
        q_var,  # process_noise_covariance
        f_func,  # state_transition_matrix(dt is a parameter)
        maha_dist_threshold=9.21,  # threshold for Mahalanobis distance
        timeout=1.0,  # timeout for initialization
        verbose=False,
    ):
        self.x_0 = x_0
        self.p = p
        self.q_var = q_var
        self.f_func = f_func
        self.maha_dist_threshold = maha_dist_threshold
        self.timeout = timeout
        self.verbose = verbose

        self.kf = KalmanFilterPy(dim_x=dim_x, dim_z=dim_z)
        self.kf.H = h
        self.kf.R = r
        self.prev_timestamp = None

    def __initialize(self):
        self.kf.P = self.p
        self.kf.x = self.x_0

    def update(self, timestamp, z=None):
        if (
            self.prev_timestamp is None
            or timestamp - self.prev_timestamp > self.timeout
        ):
            self.__initialize()
            self.prev_timestamp = timestamp

        dt = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp
        self.kf.F = self.f_func(dt)
        self.kf.Q = Q_discrete_white_noise(
            dim=self.kf.dim_x / self.kf.dim_z,
            dt=dt,
            block_size=self.kf.dim_z,
            var=self.q_var,
        )
        self.kf.predict()
        if z is not None:
            y = z - self.z_hat
            S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
            d_m2 = y.T @ np.linalg.inv(S) @ y
            # Mahalanobis distance thresholding
            if d_m2 < self.maha_dist_threshold:
                self.kf.update(z)
            else:
                if self.verbose:
                    print(f"Mahalanobis distance: {d_m2}")

    @property
    def z_hat(self):
        return self.kf.H @ self.kf.x
