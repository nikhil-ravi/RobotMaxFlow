import numpy as np
import numpy.typing as npt
import sklearn.datasets as skd
from src.Target import Target


class Robot:
    def __init__(
        self, idx: int, x: float, y: float, theta: float, view_radius: float = 5, view_angle: float = np.pi / 4
    ) -> None:
        self.idx = idx
        self.x = x
        self.y = y
        self.theta = theta
        self.observation_dim = 2
        self.H = np.hstack((np.eye(self.observation_dim), np.zeros((self.observation_dim, self.observation_dim))))
        self.Q = skd.make_spd_matrix(self.observation_dim)  # //TODO
        self.view_angle = view_angle
        self.view_radius = view_radius

    def __eq__(self, __o: object) -> bool:
        return self.idx == __o.idx and isinstance(__o, Robot)

    def __repr__(self) -> str:
        return f"Robot(id = {self.idx}, x = {self.x:.2f}, y = {self.y:.2f}, angle = {np.rad2deg(self.theta):.2f})"

    @property
    def name(self):
        return f"R{self.idx}"

    def update_state(self, control: npt.NDArray) -> npt.NDArray:
        new_state = self.state + control  # //TODO
        self.x, self.y, self.theta = new_state.flatten()
        return self.state

    def observe_target(self, target: Target) -> npt.NDArray:
        if target not in self.in_field_of_view:
            return np.zeros((self.observation_dim, 1))
        return self.H @ target.state + np.random.multivariate_normal(np.zeros(self.observation_dim), self.Q)

    @property
    def state(self) -> npt.NDArray:
        return np.atleast_2d(np.array([self.x, self.y, self.theta])).T

    def in_field_of_view(self, target: Target) -> bool:
        return (
            abs(self.relative_angle_to_target(target)) <= self.view_angle
            and self.distance_to_target(target) <= self.view_radius
        )

    def distance_to_target(self, target: Target) -> float:
        return np.sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)

    def relative_angle_to_target(self, target: Target) -> float:
        angle = self.theta - np.arctan2(target.y - self.y, target.x - self.x)
        if angle > np.pi:
            angle -= 2 * np.pi
        if angle < -np.pi:
            angle += 2 * np.pi
        return angle
