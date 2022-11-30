from collections import defaultdict
import networkx as nx
from networkx import bipartite
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


@dataclass
class Matching:
    sum_weights: float
    edges: List[Tuple[str, str]]


class RobotTargetGraph:
    def __init__(self, G: nx.Graph, add_duplicates: bool = False, seed=1234):
        np.random.seed(seed)
        self.orig_G = G.copy()
        self.robots = {n for n, d in self.orig_G.nodes(data=True) if d["bipartite"] == 0}
        self.targets = set(self.orig_G) - self.robots
        self.robots = list(self.robots)
        self.targets = list(self.targets)
        self.pos = nx.bipartite_layout(self.orig_G, nodes=self.robots)
        for key, value in self.pos.items():
            self.pos[key] = value.tolist()
        for target in self.targets:
            self.pos[target][0] -= 1
        nx.set_node_attributes(self.orig_G, self.pos, "pos")
        self.G = G.copy()
        if add_duplicates:
            self.add_duplicate_nodes()
        self.robots = {n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0}
        self.targets = set(self.G) - self.robots
        self.robots = list(self.robots)
        self.targets = list(self.targets)
        self.pos = nx.bipartite_layout(self.G, nodes=self.robots)
        for key, value in self.pos.items():
            self.pos[key] = value.tolist()
        for target in self.targets:
            self.pos[target][0] -= 1
        nx.set_node_attributes(self.G, self.pos, "pos")

    def add_duplicate_nodes(self):
        orig_robots = {n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0}
        nodeCntr = {robot: 0 for robot in orig_robots}
        self.dummy_nodes = defaultdict(list)
        for node in orig_robots:
            for (u, v, attr) in self.G.out_edges(node, data=True):
                new_node_name = f"{u}x{nodeCntr[node]}"
                self.G.add_node(new_node_name, bipartite=0)
                self.G.add_edge(new_node_name, v, **attr)
                self.dummy_nodes[node].append(new_node_name)
                nodeCntr[node] += 1
            self.G.remove_node(node)

    def set_edge_attributes(self, attrs: Dict):
        """Add edge attributes to the network

        Args:
            attrs (List[Dict]): List of attributes, where each element corresponds
            to the Dictionary of attribute values.
            names (List[str]): List of names of the attributes to be added.
        """
        for name, attr in attrs.items():
            nx.set_edge_attributes(self.G, attr, name=name)

    def _get_sum_of_weights(self, edges: List[Tuple[str, str]]) -> float:
        """Get the sum of the weights in a given matching.

        Args:
            edges (List[Tuple[str, str]]): The list of edges whose weights are to be summed.

        Returns:
            float: Sum of the weights in the given matching.
        """
        return sum([self.G[edge[0]][edge[1]]["distance"] for edge in edges])

    def build_flow_network(self, attrs: Dict) -> nx.DiGraph:
        self.flow_network = nx.from_edgelist(list(self.G.edges(data=True)), create_using=nx.DiGraph)
        """Builds the flow network of the Davis Women network.
        Convert the network into a flow network by:
            1. Convert the edges to directed edges going from the women to targets.
            2. Adding a source node `s`
            3. Adding a target node `t`
            4. Connect `s` to all the `women`
            5. Connect all the `targets` to `t`
            6. Set the utility and distance of each of the new edges to zero and their capacity to 1. 

        Args:
        attrs (Dict): Attributes dictionary. Should contain the distance and capacity keys.
        
        Returns:
            nx.DiGraph: The flow network of the Davis Women network.
        """
        self.flow_network.add_edges_from(
            [
                (
                    "s",
                    robot,
                    attrs,
                )
                for robot in self.robots
            ]
        )
        self.flow_network.add_edges_from(
            [
                (
                    target,
                    "t",
                    attrs,
                )
                for target in self.targets
            ]
        )
        pos = self.flow_layout(self.flow_network, "s", "t")
        nx.set_node_attributes(self.flow_network, pos, "pos")
        return self.flow_network

    def maximum_flow_matching(self, min_cost: bool = True, weight: str = "distance") -> Matching:
        """Get the maximum flow of the flow network of the Davis Women network.

        Args:
            min_cost (bool, optional): Whether to optimize for minimum cost flow.
            Defaults to True.
            weight (str, optional): The attribute to be used as the cost that is to be minimized.
            Defaults to "distance".

        Returns:
            Matching: The matching via maximum flow conversion.
        """
        if min_cost:
            flowDict = nx.max_flow_min_cost(self.flow_network, s="s", t="t", capacity="capacity", weight=weight)
        else:
            _, flowDict = nx.maximum_flow(self.flow_network, _s="s", _t="t", capacity="capacity")
        maxMatching = [
            (key, key2)
            for key, value in flowDict.items()
            for key2, value2 in value.items()
            if value2 and key not in ["s", "t"] and key2 not in ["s", "t"]
        ]
        sumWeights = self._get_sum_of_weights(maxMatching)
        return Matching(edges=maxMatching, sum_weights=sumWeights)

    def draw_bipartite(self, nx_kwargs: Dict = {}) -> None:
        nx.draw_networkx_nodes(
            self.G,
            nx.get_node_attributes(self.G, "pos"),
            self.robots,
            **nx_kwargs,
        )
        nx.draw_networkx_nodes(
            self.G,
            nx.get_node_attributes(self.G, "pos"),
            self.targets,
            node_size=80,
            node_color="g",
            **nx_kwargs,
        )
        nx.draw_networkx_labels(self.G, nx.get_node_attributes(self.G, "pos"), font_size=10, **nx_kwargs)
        nx.draw_networkx_edges(self.G, nx.get_node_attributes(self.G, "pos"), **nx_kwargs)

    def draw_flow_network(self, nx_kwargs: Dict = {}) -> None:
        nx.draw(
            self.flow_network,
            pos=nx.get_node_attributes(self.flow_network, "pos"),
            with_labels=True,
            **nx_kwargs,
        )

    def flow_layout(self, graph: nx.DiGraph, _s: Union[str, int] = "s", _t: Union[str, int] = "t") -> Dict:
        """Position nodes other than the source and target in two straight lines.
        The left line includes all the successors of the source node and the right line
        includes all the predecessors of the target node. The source node is then placed
        to the left of the midpoint of the first line and the target node is placed to the
        right of the midpoint of the second line.

        Args:
            graph (nx.DiGraph): The flow network that must include the passed source node _s
            and the passed target node _t.
            _s (Union[str, int], optional): The source node. Defaults to "s".
            _t (Union[str, int], optional): The target node. Defaults to "t".

        Returns:
            Dict: A dictionary of positions keyed by node.

        Examples::
            >>> G = <a flow network>
            >>> pos = flow_layout(G, "s", "t")
            >>> nx.draw(G, pos=pos)

        """
        pos = nx.bipartite_layout(
            self.robots + self.targets,
            graph.successors(_s),
        )
        # t centers
        s_center = np.mean([pos[i][1] for i in graph.successors(_s)])
        t_center = np.mean([pos[i][1] for i in graph.predecessors(_t)])
        diff = t_center - s_center
        for i in graph.predecessors(_t):
            pos[i][1] -= diff
        t_center -= diff
        pos[_s] = [np.mean([pos[i][0] for i in graph.successors(_s)]) - 1, s_center]
        pos[_t] = [np.mean([pos[i][0] for i in graph.predecessors(_t)]) + 1, t_center]
        return pos

    def draw_flow(
        self,
        matching: Matching,
        contract: bool = True,
        node_kwargs: Dict = {},
        label_kwargs: Dict = {},
    ) -> None:
        """Draw the flow network highlighting the flows.

        Args:
            graph (nx.DiGraph): The flow graph.
            matching (Matching): The Matching.
            nx_kwargs (Dict, optional): Extra arguments to control the network figures. Defaults to {}.
        """
        H = self.G
        edgelist = matching.edges
        if contract:
            # H = self.G.copy()
            # for node in self.dummy_nodes.keys():
            #     for dummy_node in self.dummy_nodes[node]:
            #         H = nx.contracted_nodes(H, node, dummy_node)
            H = self.orig_G
            contracted_edgelist = []
            for u, v in edgelist:
                contracted_edgelist.append((u.split("x")[0], v))
            edgelist = contracted_edgelist
        dpos = nx.get_node_attributes(H, "pos")
        nx.draw_networkx_nodes(
            H,
            pos=dpos,
            **node_kwargs,
        )
        nx.draw_networkx_labels(H, pos=dpos, **label_kwargs)
        nx.draw_networkx_edges(
            H,
            pos=dpos,
            edge_color="k",
            alpha=0.5,
        )
        nx.draw_networkx_edges(H, pos=dpos, edgelist=edgelist, edge_color="r")
        plt.box(False)
