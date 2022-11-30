import numpy as np
import numpy.typing as npt
import sklearn.datasets as skd


class Target:
    def __init__(self, idx: int, x: float, y: float, vel_x: float, vel_y: float) -> None:
        self.idx = idx
        self.x = x
        self.y = y
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.state_dim = 4
        self.F = np.random.random((self.state_dim, 1))  # //TODO
        self.Q = skd.make_spd_matrix(self.state_dim)  # //TODO

    def __eq__(self, __o: object) -> bool:
        return self.idx == __o.idx and isinstance(__o, Target)

    def __repr__(self) -> str:
        return (
            f"Target(id = {self.idx}, x = {self.x:.2f}, y = {self.y:.2f}, vx = {self.vel_x:.2f}, vy = {self.vel_y:.2f})"
        )

    @property
    def name(self):
        return f"T{self.idx}"

    def update_state(self) -> npt.NDArray:
        self.x, self.y, self.vel_x, self.vel_y = (self.F @ self.state).flatten() + np.random.multivariate_normal(
            np.zeros(self.state_dim), self.Q
        )
        return self.state

    @property
    def state(self) -> npt.NDArray:
        return np.atleast_2d(np.array([self.x, self.y, self.vel_x, self.vel_y])).T
