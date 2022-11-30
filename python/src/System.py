import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from src.Target import Target
from src.Robot import Robot
from src.RobotTargetGraph import RobotTargetGraph


class System:
    def __init__(self, N: int, M: int) -> None:
        self.N = N
        self.M = M
        np.random.seed(5678)
        self.init_robots()
        self.init_targets()

    def init_robots(self, init_pos: list = None, init_thetas: list = None):
        self.robots = []
        if not init_pos:
            init_pos = []
            for idx in range(self.N):
                init_pos.append(np.random.uniform(-10, 10, (2,)).tolist())
        if not init_thetas:
            init_thetas = []
            for idx in range(self.N):
                init_thetas.append(np.random.uniform(0, 2 * np.pi))
        for idx in range(self.N):
            self.robots.append(Robot(idx, *init_pos[idx], init_thetas[idx]))

    def init_targets(self, init_pos: list = None, init_vels: list = None):
        self.targets = []
        if not init_pos:
            init_pos = []
            for idx in range(self.M):
                init_pos.append(np.random.uniform(-10, 10, (2,)).tolist())
        if not init_vels:
            init_vels = []
            for idx in range(self.M):
                init_vels.append(np.random.uniform(-2, 2, (2,)).tolist())
        for idx in range(self.M):
            self.targets.append(Target(idx, *init_pos[idx], *init_vels[idx]))

    def fields_of_view(self):
        edges = []
        for robot in self.robots:
            for target in self.targets:
                if robot.in_field_of_view(target):
                    edges.append((robot.name, target.name, {"distance": robot.distance_to_target(target)}))
        return edges

    @property
    def G(self):
        G = nx.DiGraph()
        G.add_nodes_from([robot.name for robot in self.robots], bipartite=0)
        G.add_nodes_from([target.name for target in self.targets], bipartite=1)
        G.add_edges_from(self.fields_of_view())
        return G

    def plot_system(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        ax = axs[0]
        for robot in self.robots:
            ax.plot(robot.x, robot.y, marker="s", c="k")
            ax.add_artist(
                Wedge(
                    (robot.x, robot.y),
                    robot.view_radius,
                    np.rad2deg(robot.theta),
                    np.rad2deg(robot.theta + robot.view_angle),
                    edgecolor="b",
                    facecolor="none",
                    alpha=0.1,
                )
            )
            ax.add_artist(
                Wedge(
                    (robot.x, robot.y),
                    robot.view_radius,
                    np.rad2deg(robot.theta - robot.view_angle),
                    np.rad2deg(robot.theta),
                    edgecolor="b",
                    facecolor="none",
                    alpha=0.1,
                )
            )
            ax.annotate(robot.name, (robot.x, robot.y))
        for target in self.targets:
            ax.plot(target.x, target.y, marker="x", c="r", markersize=5)
            ax.annotate(target.name, (target.x, target.y), fontsize=5)

        # Draw bipartite graph
        RobotTargetGraph(self.G).draw_bipartite({"ax": axs[1]})
        plt.show()
