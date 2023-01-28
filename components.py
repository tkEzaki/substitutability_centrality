import numpy as np
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse import csr_matrix
import random
import math
import pickle
import pandas as pd
import networkx as nx


np.set_printoptions(threshold=np.inf)

EPSILON = 0.0001


class Network:
    def __init__(self):
        self.link_usage_record = np.zeros((self.node_num, self.node_num))
        self.node_usage_record = np.zeros(self.node_num)
        self.link_usage_record_recent = np.zeros((self.node_num, self.node_num))
        self.node_usage_record_recent = np.zeros(self.node_num)
        self.mixture_ratio = None
        self.network_name = "network"
        self.capacity_coef = None
        self.capacity_type = None

    def set_adjacency_matrix_uniform(self, range_width=None):
        self.adjacency_matrix = np.zeros((self.node_num, self.node_num))
        self.connectivity_matrix = np.zeros((self.node_num, self.node_num))
        self.link_list = []
        if range_width is None:
            range_width = 1

        for i in range(self.node_num):
            for j in self.nodes[i].neighbors:
                self.adjacency_matrix[i][j] = 1 + range_width * (np.random.rand() - 0.5)
                self.connectivity_matrix[i][j] = 1
                self.link_list.append((i, j))

    def set_generate_pattern_random_uniform(self):
        self.generate_pattern = [np.random.rand(self.node_num), np.random.rand(self.node_num)]
        print("----generate_pattern generated", self.generate_pattern)

    def set_shortest_path_and_link_bw_centrality(self, set_od_matrix=True):
        if set_od_matrix:  # self.od_demand_matrix is not yet set
            if self.generate_pattern is None:
                self.od_demand_matrix = np.ones(self.adjacency_matrix.shape) / self.adjacency_matrix.size
            else:
                generate_orig = np.array(self.generate_pattern[0]) / np.sum(self.generate_pattern[0])
                generate_dest = np.array(self.generate_pattern[1]) / np.sum(self.generate_pattern[1])
                self.od_demand_matrix = generate_orig.reshape(-1, 1) @ generate_dest.reshape(1, -1)

        for i in range(len(self.nodes)):
            self.nodes[i].shortest_paths = []  # initialize

        self.link_bw_centrality, self.link_bw_centrality_demand = \
            self.compute_link_bw_centrality(self.adjacency_matrix,
                                            associate_shortest_paths=True, allow_disconnection=False)

        return

    def set_substitutability_centrality(self):
        self.substitutability_centrality = np.zeros(self.link_bw_centrality.shape)
        self.substitutability_centrality_demand = np.zeros(self.link_bw_centrality.shape)
        num_links = 0

        for i in range(self.node_num):
            for j in self. nodes[i].neighbors:
                num_links += 1
                partial_adjacency_matrix = self.adjacency_matrix.copy()
                partial_adjacency_matrix[i][j] = 0  # remove a link!
                n, labels = connected_components(csr_matrix(partial_adjacency_matrix), connection='strong')
                if n == 1:
                    partial_link_bw_centrality, partial_link_bw_centrality_demand = \
                        self.compute_link_bw_centrality(partial_adjacency_matrix)
                    """ count only positive contribution"""
                    idx = self.link_bw_centrality < partial_link_bw_centrality
                    idx_demand = self.link_bw_centrality_demand < partial_link_bw_centrality_demand
                    self.substitutability_centrality[idx] += partial_link_bw_centrality[idx] - self.link_bw_centrality[idx]
                    self.substitutability_centrality_demand[idx_demand] += partial_link_bw_centrality_demand[idx_demand] - self.link_bw_centrality_demand[idx_demand]
                    """"""
                    # count all contributions
                    # self.substitutability_centrality += partial_link_bw_centrality - self.link_bw_centrality
                    # self.substitutability_centrality_demand += partial_link_bw_centrality_demand - self.link_bw_centrality_demand

        self.substitutability_centrality /= num_links
        self.substitutability_centrality_demand /= num_links

    def compute_link_bw_centrality(self, adjacency_matrix, associate_shortest_paths=False, allow_disconnection=True):
        _, last_visited = shortest_path(csr_matrix(adjacency_matrix), directed=True, return_predecessors=True)
        link_bw_centrality = np.zeros(adjacency_matrix.shape)
        link_bw_centrality_demand = np.zeros(adjacency_matrix.shape)

        for i in range(self.node_num):  # origin
            for j in range(self.node_num):  # destination
                if i == j:
                    if associate_shortest_paths:
                        self.nodes[i].shortest_paths.append([i, i])
                else:
                    current_node = j
                    path = []
                    for k in range(self.node_num):
                        # update from the destination in the reversed direction
                        previous_node = last_visited[i][current_node]
                        if previous_node == -9999:
                            if allow_disconnection:
                                print("disconnected!")
                                break
                            else:
                                raise Exception("disconnected!")
                        link_bw_centrality[previous_node][current_node] += 1
                        link_bw_centrality_demand[previous_node][current_node] += self.od_demand_matrix[i][j]
                        path.append(current_node)
                        current_node = previous_node
                        if current_node == i:  # fin
                            path.append(current_node)
                            if associate_shortest_paths:
                                self.nodes[i].shortest_paths.append(list(reversed(path)))
                            break

        return link_bw_centrality, link_bw_centrality_demand

    def set_capacity_dist(self, capacity_type="PPL", capacity_coef=0):
        if capacity_type == "WBC":
            self.link_capacity = self.connectivity_matrix.copy() \
                + self.link_bw_centrality * capacity_coef * np.sum(self.connectivity_matrix) \
                / np.sum(self.link_bw_centrality)
        elif capacity_type == "PPL":
            self.link_capacity = self.connectivity_matrix.copy() * (1 + capacity_coef)
        else:
            print("capacity_type: {} unknown.".format(capacity_type))

        self.capacity_type = capacity_type
        self.capacity_coef = capacity_coef  # for record

    def set_capacity_dist_substitutability(self, capacity_coef=1, mixture_ratio=1, consider_demand=False):
        if mixture_ratio < 0 or mixture_ratio > 1:
            raise Exception("Invalid mixture ratio!")

        self.mixture_ratio = mixture_ratio
        if consider_demand:
            substitutability_capacity = self.connectivity_matrix.copy() \
                + self.substitutability_centrality_demand * capacity_coef * np.sum(self.connectivity_matrix)\
                / np.sum(self.substitutability_centrality_demand)
            betweenness_capacity = self.connectivity_matrix.copy() \
                + self.link_bw_centrality_demand * capacity_coef * np.sum(self.connectivity_matrix)\
                / np.sum(self.link_bw_centrality_demand)

            self.link_capacity = mixture_ratio * substitutability_capacity + (1 - mixture_ratio) * betweenness_capacity

        else:
            substitutability_capacity = self.connectivity_matrix.copy() \
                + self.substitutability_centrality * capacity_coef * np.sum(self.connectivity_matrix) \
                / np.sum(self.substitutability_centrality)
            betweenness_capacity = self.connectivity_matrix.copy() \
                + self.link_bw_centrality * capacity_coef * np.sum(self.connectivity_matrix) \
                / np.sum(self.link_bw_centrality)

            self.link_capacity = mixture_ratio * substitutability_capacity + (1 - mixture_ratio) * betweenness_capacity

        self.capacity_coef = capacity_coef
        self.capacity_type = "WSC_demand" if consider_demand else "WSC"
        if mixture_ratio == 0:
            self.capacity_type = "WBC_demand" if consider_demand else "WBC"


class ConnectedHubAndSpokeNetwork(Network):
    def __init__(self, leaf_num=5, capacity_type="PPL", capacity_coef=0,
                 generate_pattern=None, binary_cost=False, c_plus=2):
        self.set_connected_hub_and_spoke(leaf_num)
        self.core_links = [
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (0, 3), (3, 0),
            (1, 2), (2, 1),
            (1, 3), (3, 1),
            (2, 3), (3, 2),
        ]
        self.bypass_core_links = [
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (0, 3), (3, 0),
            (1, 2), (2, 1),
            (1, 3), (3, 1),
            (2, 3), (3, 2),
        ]
        super().__init__()
        self.generate_pattern = generate_pattern
        if binary_cost:
            self.set_adjacency_matrix_binary_core_periphery(c_plus=c_plus)
        else:
            super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_bw_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.network_name = "connected_hub_and_spoke"

    def set_adjacency_matrix_binary_core_periphery(self, c_plus=2):
        # set cij = 1 for periphery links and cij = c_plus for core links
        self.adjacency_matrix = np.zeros((self.node_num, self.node_num))
        self.connectivity_matrix = np.zeros((self.node_num, self.node_num))
        self.link_list = []
        for i in range(self.node_num):
            for j in self.nodes[i].neighbors:
                if (i, j) in self.bypass_core_links:
                    self.adjacency_matrix[i][j] = c_plus
                else:
                    self.adjacency_matrix[i][j] = 1

                self.connectivity_matrix[i][j] = 1
                self.link_list.append((i, j))

    def set_connected_hub_and_spoke(self, leaf_num=5):
        self.node_num = 4 + 4 * leaf_num
        self.nodes = [Node() for i in range(self.node_num)]

        for i in range(1, 5):
            self.nodes[i - 1].neighbors = [4 + (i - 1) * leaf_num + j for j in range(leaf_num)]
            self.nodes[i - 1].x, self.nodes[i - 1].y = \
                0.25 * math.cos(math.pi * 0.5 * i), 0.25 * math.sin(math.pi * 0.5 * i)

            for j in range(leaf_num):
                self.nodes[4 + (i - 1) * leaf_num + j].x, self.nodes[4 + (i - 1) * leaf_num + j].y = \
                    0.25 * math.cos(math.pi * 0.5 * i) + 0.25 * math.cos(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1)),\
                    0.25 * math.sin(math.pi * 0.5 * i) + 0.25 * math.sin(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1))

                self.nodes[4 + (i - 1) * leaf_num + j].neighbors.append(i - 1)

        self.nodes[0].neighbors.extend([1, 2, 3])
        self.nodes[1].neighbors.extend([0, 2, 3])
        self.nodes[2].neighbors.extend([0, 1, 3])
        self.nodes[3].neighbors.extend([0, 1, 2])

    def set_generate_pattern_distorted(self, w=2):
        origin_pattern = [1 for i in range(self.node_num)]
        destination_pattern = [1 for i in range(self.node_num)]
        origin_pattern[0] = 0
        origin_pattern[1] = 0
        origin_pattern[2] = 0
        origin_pattern[3] = 0

        destination_pattern[0] = 0
        destination_pattern[1] = 0
        destination_pattern[2] = 0
        destination_pattern[3] = 0

        origin_pattern[19] = w
        origin_pattern[20] = w
        origin_pattern[21] = w
        origin_pattern[22] = w
        origin_pattern[23] = w

        destination_pattern[14] = w
        destination_pattern[15] = w
        destination_pattern[16] = w
        destination_pattern[17] = w
        destination_pattern[18] = w

        self.generate_pattern = [origin_pattern, destination_pattern]


class GeneralHubAndSpokeNetwork(Network):
    def __init__(self, hub_num=4, leaf_num=5, capacity_type="PPL", capacity_coef=0,
                 generate_pattern=None):
        self.set_general_hub_and_spoke(hub_num, leaf_num)
        self.core_links = [
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (0, 3), (3, 0),
            (1, 2), (2, 1),
            (1, 3), (3, 1),
            (2, 3), (3, 2),
        ]
        self.bypass_core_links = [
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (0, 3), (3, 0),
            (1, 2), (2, 1),
            (1, 3), (3, 1),
            (2, 3), (3, 2),
        ]
        super().__init__()
        self.generate_pattern = generate_pattern
        super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_bw_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.network_name = "general_hub_and_spoke"

    def set_general_hub_and_spoke(self, hub_num=4, leaf_num=5):
        self.node_num = hub_num + hub_num * leaf_num
        self.nodes = [Node() for i in range(self.node_num)]
        regions = []
        region_count = 0
        leaf_angle = math.pi / 3
        for hub in range(hub_num):
            regions.append(
                {
                    "hub_node": (1 + leaf_num) * region_count,
                    "leaf_nodes": [1 + i + (leaf_num + 1) * region_count for i in range(leaf_num)],
                    "hub_angle": 2 * math.pi / hub_num * region_count,
                    "leaf_angles": [2 * math.pi / hub_num * region_count - leaf_angle / 2 + i * leaf_angle / (leaf_num - 1) for i in range(leaf_num)]
                }
            )
            region_count += 1

        for region_num in range(hub_num):
            self.nodes[regions[region_num]["hub_node"]].neighbors = regions[region_num]["leaf_nodes"].copy()
            for leaf_node in regions[region_num]["leaf_nodes"]:
                self.nodes[leaf_node].neighbors.append(regions[region_num]["hub_node"])

            for other_region_num in range(hub_num):
                if region_num != other_region_num:
                    self.nodes[regions[region_num]["hub_node"]].neighbors.append(regions[other_region_num]["hub_node"])

        for region_num in range(hub_num):
            self.nodes[regions[region_num]["hub_node"]].x, self.nodes[regions[region_num]["hub_node"]].y =\
                0.4 * math.cos(regions[region_num]["hub_angle"]), 0.4 * math.sin(regions[region_num]["hub_angle"])
            for idx, leaf_node in enumerate(regions[region_num]["leaf_nodes"]):
                self.nodes[leaf_node].x, self.nodes[leaf_node].y =\
                    self.nodes[regions[region_num]["hub_node"]].x + 0.1 * math.cos(regions[region_num]["leaf_angles"][idx]),\
                    self.nodes[regions[region_num]["hub_node"]].y + 0.1 * math.sin(regions[region_num]["leaf_angles"][idx])

    def set_generate_pattern_distorted(self, w=2):
        origin_pattern = [1 for i in range(self.node_num)]
        destination_pattern = [1 for i in range(self.node_num)]
        origin_pattern[0] = 0
        origin_pattern[1] = 0
        origin_pattern[2] = 0
        origin_pattern[3] = 0

        destination_pattern[0] = 0
        destination_pattern[1] = 0
        destination_pattern[2] = 0
        destination_pattern[3] = 0

        origin_pattern[19] = w
        origin_pattern[20] = w
        origin_pattern[21] = w
        origin_pattern[22] = w
        origin_pattern[23] = w

        destination_pattern[14] = w
        destination_pattern[15] = w
        destination_pattern[16] = w
        destination_pattern[17] = w
        destination_pattern[18] = w

        self.generate_pattern = [origin_pattern, destination_pattern]


class ReinforcedHubAndSpokeNetwork(Network):
    def __init__(self, leaf_num=5, capacity_type="PPL", capacity_coef=0,
                 generate_pattern=None, binary_cost=False, c_plus=2):
        self.set_reinforced_hub_and_spoke(leaf_num)
        self.core_links = [
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (0, 3), (3, 0),
            (0, 4), (4, 0),
            (1, 2), (2, 1),
            (1, 4), (4, 1),
            (2, 3), (3, 2),
            (3, 4), (4, 3),
        ]
        self.bypass_core_links = [
            (1, 2), (2, 1),
            (1, 4), (4, 1),
            (2, 3), (3, 2),
            (3, 4), (4, 3),
        ]
        super().__init__()
        self.generate_pattern = generate_pattern
        if binary_cost:
            self.set_adjacency_matrix_binary_core_periphery(c_plus=c_plus)
        else:
            super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_bw_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.network_name = "reinforced_hub_and_spoke"

    def set_adjacency_matrix_binary_core_periphery(self, c_plus=2):
        # set cij = 1 for periphery links and cij = c_plus for core links
        self.adjacency_matrix = np.zeros((self.node_num, self.node_num))
        self.connectivity_matrix = np.zeros((self.node_num, self.node_num))
        self.link_list = []
        for i in range(self.node_num):
            for j in self.nodes[i].neighbors:
                if (i, j) in self.bypass_core_links:
                    self.adjacency_matrix[i][j] = c_plus
                else:
                    self.adjacency_matrix[i][j] = 1

                self.connectivity_matrix[i][j] = 1
                self.link_list.append((i, j))

    def set_reinforced_hub_and_spoke(self, leaf_num=5):
        self.node_num = 5 + 4 * leaf_num
        self.nodes = [Node() for i in range(self.node_num)]

        self.nodes[0].x, self.nodes[0].y = 0, 0
        self.nodes[0].neighbors = [1, 2, 3, 4]  # center

        for i in range(1, 5):
            self.nodes[i].neighbors = [5 + (i - 1) * leaf_num + j for j in range(leaf_num)]
            self.nodes[i].x, self.nodes[i].y = \
                0.25 * math.cos(math.pi * 0.5 * i), 0.25 * math.sin(math.pi * 0.5 * i)

            self.nodes[i].neighbors.append(0)
            for j in range(leaf_num):
                self.nodes[5 + (i - 1) * leaf_num + j].x, self.nodes[5 + (i - 1) * leaf_num + j].y = \
                    0.25 * math.cos(math.pi * 0.5 * i) + 0.25 * math.cos(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1)),\
                    0.25 * math.sin(math.pi * 0.5 * i) + 0.25 * math.sin(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1))

                self.nodes[5 + (i - 1) * leaf_num + j].neighbors.append(i)

        self.nodes[1].neighbors.extend([2, 4])
        self.nodes[2].neighbors.extend([1, 3])
        self.nodes[3].neighbors.extend([2, 4])
        self.nodes[4].neighbors.extend([3, 1])


class ReinforcedHubAndSpokeNetworkSquare(Network):
    def __init__(self, leaf_num=5, capacity_type="PPL", capacity_coef=0, generate_pattern=None, binary_cost=False, c_plus=2):
        self.set_reinforced_hub_and_spoke_square(leaf_num)
        self.core_links = [
            (0, 1), (1, 0),
            (0, 2), (2, 0),
            (0, 3), (3, 0),
            (0, 4), (4, 0),
            (25, 1), (1, 25),
            (25, 4), (4, 25),
            (26, 1), (1, 26),
            (26, 2), (2, 26),
            (27, 2), (2, 27),
            (27, 3), (3, 27),
            (28, 3), (3, 28),
            (28, 4), (4, 28),
        ]
        self.bypass_core_links = [
            (25, 1), (1, 25),
            (25, 4), (4, 25),
            (26, 1), (1, 26),
            (26, 2), (2, 26),
            (27, 2), (2, 27),
            (27, 3), (3, 27),
            (28, 3), (3, 28),
            (28, 4), (4, 28),
        ]
        super().__init__()
        self.generate_pattern = generate_pattern
        if binary_cost:
            self.set_adjacency_matrix_binary_core_periphery(c_plus=c_plus)
        else:
            super().set_adjacency_matrix_uniform()

        super().set_shortest_path_and_link_bw_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.network_name = "reinforced_hub_and_spoke_square"

    def set_adjacency_matrix_binary_core_periphery(self, c_plus=2):
        # set cij = 1 for periphery links and cij = c_plus for core links
        self.adjacency_matrix = np.zeros((self.node_num, self.node_num))
        self.connectivity_matrix = np.zeros((self.node_num, self.node_num))
        self.link_list = []
        for i in range(self.node_num):
            for j in self.nodes[i].neighbors:
                if (i, j) in self.bypass_core_links:
                    self.adjacency_matrix[i][j] = c_plus
                else:
                    self.adjacency_matrix[i][j] = 1

                self.connectivity_matrix[i][j] = 1
                self.link_list.append((i, j))

    def set_reinforced_hub_and_spoke_square(self, leaf_num=5):
        self.node_num = 5 + 4 * leaf_num + 4
        self.nodes = [Node() for i in range(self.node_num)]

        self.nodes[0].x, self.nodes[0].y = 0, 0

        self.nodes[0].neighbors = [1, 2, 3, 4]  # center

        for i in range(1, 5):
            self.nodes[i].neighbors = [5 + (i - 1) * leaf_num + j for j in range(leaf_num)]
            self.nodes[i].x, self.nodes[i].y = \
                0.25 * math.cos(math.pi * 0.5 * i), 0.25 * math.sin(math.pi * 0.5 * i)

            self.nodes[i].neighbors.append(0)
            for j in range(leaf_num):
                self.nodes[5 + (i - 1) * leaf_num + j].x, self.nodes[5 + (i - 1) * leaf_num + j].y = \
                    0.25 * math.cos(math.pi * 0.5 * i) + 0.25 * math.cos(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1)), \
                    0.25 * math.sin(math.pi * 0.5 * i) + 0.25 * math.sin(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1))

                self.nodes[5 + (i - 1) * leaf_num + j].neighbors.append(i)

        self.nodes[25].x, self.nodes[25].y = 0.25, 0.25
        self.nodes[26].x, self.nodes[26].y = -0.25, 0.25
        self.nodes[27].x, self.nodes[27].y = -0.25, -0.25
        self.nodes[28].x, self.nodes[28].y = 0.25, -0.25

        self.nodes[25].neighbors = [1, 4]
        self.nodes[26].neighbors = [1, 2]
        self.nodes[27].neighbors = [2, 3]
        self.nodes[28].neighbors = [3, 4]

        self.nodes[1].neighbors.extend([25, 26])
        self.nodes[2].neighbors.extend([26, 27])
        self.nodes[3].neighbors.extend([27, 28])
        self.nodes[4].neighbors.extend([28, 25])


class HubAndSpokeNetwork(Network):
    def __init__(self, leaf_num=5, capacity_type="PPL", capacity_coef=0, generate_pattern=None):
        self.set_hub_and_spoke(leaf_num)
        self.generate_pattern = generate_pattern
        super().__init__()
        super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_bw_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.network_name = "hub_and_spoke"

    def set_hub_and_spoke(self, leaf_num=5):
        self.node_num = 5 + 4 * leaf_num
        self.nodes = [Node() for i in range(self.node_num)]

        self.nodes[0].x, self.nodes[0].y = 0, 0

        self.nodes[0].neighbors = [1, 2, 3, 4]  # center

        for i in range(1, 5):
            self.nodes[i].neighbors = [5 + (i - 1) * leaf_num + j for j in range(leaf_num)]
            self.nodes[i].x, self.nodes[i].y = \
                0.25 * math.cos(math.pi * 0.5 * i), 0.25 * math.sin(math.pi * 0.5 * i)

            self.nodes[i].neighbors.append(0)
            for j in range(leaf_num):
                self.nodes[5 + (i - 1) * leaf_num + j].x, self.nodes[5 + (i - 1) * leaf_num + j].y = \
                    0.25 * math.cos(math.pi * 0.5 * i) + 0.25 * math.cos(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1)), \
                    0.25 * math.sin(math.pi * 0.5 * i) + 0.25 * math.sin(
                        math.pi * 0.5 * i - math.pi * 0.5 + math.pi / (leaf_num + 1) * (j + 1))

                self.nodes[5 + (i - 1) * leaf_num + j].neighbors.append(i)


class LatticeNetwork(Network):
    def __init__(self, L: int, capacity_type="PPL", capacity_coef=0, generate_pattern=None):
        self.set_lattice(L)
        self.generate_pattern = generate_pattern
        super().__init__()
        super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_bw_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.network_name = "square_lattice"

    def set_lattice(self, L):
        self.node_num = L * L
        self.nodes = [Node() for i in range(self.node_num)]

        for i in range(L):
            for j in range(L):
                self.nodes[i * L + j].x, self.nodes[i * L + j].y \
                    = - 0.5 + 1 / (L - 1) * i, 0.5 - 1 / (L - 1) * j

        # bulk
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                node_id_center = i * L + j
                node_id_left = i * L + j - 1
                node_id_right = i * L + j + 1
                node_id_up = (i - 1) * L + j
                node_id_down = (i + 1) * L + j
                self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]

        # top boundary
        for j in range(1, L - 1):
            node_id_center = 0 * L + j
            node_id_left = 0 * L + j - 1
            node_id_right = 0 * L + j + 1
            # node_id_up = (L - 1) * L + j
            node_id_down = 1 * L + j
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_down]

        # down boundary
        for j in range(1, L - 1):
            node_id_center = (L - 1) * L + j
            node_id_left = (L - 1) * L + j - 1
            node_id_right = (L - 1) * L + j + 1
            node_id_up = (L - 2) * L + j
            # node_id_down = 0 * L + j
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up]

        # left boundary
        for i in range(1, L - 1):
            node_id_center = i * L + 0
            # node_id_left = i * L + L - 1
            node_id_right = i * L + 0 + 1
            node_id_up = (i - 1) * L + 0
            node_id_down = (i + 1) * L + 0
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_right, node_id_up, node_id_down]

        # right boundary
        for i in range(1, L - 1):
            node_id_center = i * L + (L - 1)
            node_id_left = i * L + (L - 1) - 1
            # node_id_right = i * L + 0
            node_id_up = (i - 1) * L + (L - 1)
            node_id_down = (i + 1) * L + (L - 1)
            # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
            self.nodes[node_id_center].neighbors = [node_id_left, node_id_up, node_id_down]

        # left up corner
        node_id_center = 0
        # node_id_left = (L - 1)
        node_id_right = 1
        # node_id_up = (L - 1) * L
        node_id_down = 1 * L
        # self.nodes[node_id_center].neighbors = [node_id_left, node_id_right, node_id_up, node_id_down]
        self.nodes[node_id_center].neighbors = [node_id_right, node_id_down]

        # left down corner
        node_id_center = (L - 1) * L
        # node_id_left = (L - 1) * L + L - 1
        node_id_right = (L - 1) * L + 1
        node_id_up = (L - 2) * L
        # node_id_down = 0
        self.nodes[node_id_center].neighbors = [node_id_right, node_id_up]

        # right up corner
        node_id_center = L - 1
        node_id_left = L - 2
        # node_id_right = 0
        # node_id_up = (L - 1) * L + L - 1
        node_id_down = 1 * L + L - 1
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_down]

        # right down corner
        node_id_center = (L - 1) * L + L - 1
        node_id_left = (L - 1) * L + L - 2
        # node_id_right = (L - 1) * L
        node_id_up = (L - 2) * L + L - 1
        # node_id_down = 1 * L - 1
        self.nodes[node_id_center].neighbors = [node_id_left, node_id_up]

    def set_generate_pattern_distorted(self, w=2):
        origin_pattern = [1 for i in range(self.node_num)]
        destination_pattern = [1 for i in range(self.node_num)]

        origin_pattern[0] = w
        origin_pattern[1] = w
        origin_pattern[2] = w
        origin_pattern[3] = w
        origin_pattern[4] = w

        destination_pattern[self.node_num - 5] = w
        destination_pattern[self.node_num - 4] = w
        destination_pattern[self.node_num - 3] = w
        destination_pattern[self.node_num - 2] = w
        destination_pattern[self.node_num - 1] = w

        self.generate_pattern = [origin_pattern, destination_pattern]


class RandomNetwork(Network):
    def __init__(self, node_num, link_num=40, capacity_type="PPL", capacity_coef=0, generate_pattern=None):
        self.set_random_net(node_num, link_num)
        self.generate_pattern = generate_pattern
        super().__init__()
        super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_bw_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.network_name = "random_network"

    def set_random_net(self, node_num, link_num=40):
        if node_num > link_num:
            print("Warning! Random network is not connected!")
        elif (node_num - 1) * node_num / 2 < link_num:
            print("Warning! Too many links in random network.")

        self.node_num = node_num
        self.nodes = [Node() for i in range(self.node_num)]

        for i in range(self.node_num):
            self.nodes[i].x, self.nodes[i].y = \
                0.5 * math.cos(2 * math.pi / self.node_num * i), 0.5 * math.sin(2 * math.pi / self.node_num * i)

        neighbors_list = [[] for i in range(self.node_num)]
        # at least 1 link per node
        for i in range(self.node_num):
            selected_node = random.randrange(self.node_num)
            while selected_node == i or i in neighbors_list[selected_node]:
                selected_node = random.randrange(self.node_num)

            neighbors_list[i].append(selected_node)
            neighbors_list[selected_node].append(i)

        for link in range(link_num - self.node_num):
            selected_node_1 = random.randrange(self.node_num)
            selected_node_2 = random.randrange(self.node_num)
            while selected_node_1 == selected_node_2 \
                    or selected_node_2 in neighbors_list[selected_node_1]:
                selected_node_1 = random.randrange(self.node_num)

            neighbors_list[selected_node_1].append(selected_node_2)
            neighbors_list[selected_node_2].append(selected_node_1)

        for i in range(self.node_num):
            self.nodes[i].neighbors = neighbors_list[i].copy()
        # not yet developed

        return

    def set_generate_pattern_distorted(self, w=2):
        origin_pattern = [1 for i in range(self.node_num)]
        destination_pattern = [1 for i in range(self.node_num)]

        origin_pattern[0] = w
        origin_pattern[1] = w
        origin_pattern[2] = w
        origin_pattern[3] = w
        origin_pattern[4] = w

        destination_pattern[5] = w
        destination_pattern[6] = w
        destination_pattern[7] = w
        destination_pattern[8] = w
        destination_pattern[9] = w

        self.generate_pattern = [origin_pattern, destination_pattern]


class BANetwork(Network):
    def __init__(self, node_num, link_num_each=2, capacity_type="PPL", capacity_coef=0, generate_pattern=None):
        self.set_BA_net(node_num, link_num_each)
        self.generate_pattern = generate_pattern
        super().__init__()
        super().set_adjacency_matrix_uniform()
        super().set_shortest_path_and_link_bw_centrality()
        super().set_capacity_dist(capacity_type, capacity_coef)
        self.network_name = "BA_network"

    def set_BA_net(self, node_num, link_num_each):
        ba_graph = nx.barabasi_albert_graph(node_num, link_num_each)
        pos = nx.kamada_kawai_layout(ba_graph)  # set node positions

        self.node_num = node_num
        self.nodes = [Node() for i in range(self.node_num)]

        for i in range(self.node_num):
            self.nodes[i].x, self.nodes[i].y = pos[i][0], pos[i][1]

        links = ba_graph.edges()
        neighbors_list = [[] for i in range(self.node_num)]
        for link in links:
            neighbors_list[link[0]].append(link[1])
            neighbors_list[link[1]].append(link[0])

        for i in range(self.node_num):
            self.nodes[i].neighbors = neighbors_list[i].copy()

        return

    def set_generate_pattern_distorted(self, w=2):
        origin_pattern = [1 for i in range(self.node_num)]
        destination_pattern = [1 for i in range(self.node_num)]

        origin_pattern[0] = w
        origin_pattern[1] = w
        origin_pattern[2] = w
        origin_pattern[3] = w
        origin_pattern[4] = w

        destination_pattern[5] = w
        destination_pattern[6] = w
        destination_pattern[7] = w
        destination_pattern[8] = w
        destination_pattern[9] = w

        self.generate_pattern = [origin_pattern, destination_pattern]


class Node:
    neighbors = []
    quantity = 0
    shortest_paths = []
    current_packets = []

    def __init__(self):
        self.neighbors = []
        self.quantity = 0
        self.shortest_paths = []
        self.current_packet = []
        self.x = 0
        self.y = 0


class Packet:
    def __init__(self, packet_type, origin, destination, packet_id, path_plan=[], distance=0, steps=0, total_cost=0, is_traced=False,
                 trace_id=-1):
        self.packet_type = packet_type
        self.origin = origin
        self.destination = destination
        self.path_plan = path_plan
        self.current_location = origin
        self.packet_id = packet_id
        self.distance = distance
        self.remaining_path_length = len(path_plan)
        self.shortest_remaining_path_length = self.remaining_path_length
        self.steps = steps
        self.total_cost = total_cost
        self.is_traced = is_traced
        self.trace_id = trace_id

    def update_remaining_path_length(self):
        self.remaining_path_length = len(self.path_plan)


class PathPlanner:
    def __init__(self, network: Network, waiting_cost: float):
        self.path_usage = []
        self.blocked_path = []
        self.static_network = network
        self.disturbed_links = []
        self.waiting_cost = waiting_cost

    def add_path_plan(self, path):
        for i in range(len(path) - 1):
            if len(self.path_usage) > i:
                self.path_usage[i].append((path[i], path[i + 1]))
            else:
                self.path_usage.append([])
                self.path_usage[i].append((path[i], path[i + 1]))
        self.compute_blocked_path()  # compute blocked path

    def compute_blocked_path(self):
        new_blocked_path = []
        for t in range(len(self.path_usage)):
            new_capacity_matrix = self.static_network.link_capacity.copy()
            blocked_links = []
            for link in self.path_usage[t]:
                new_capacity_matrix[link[0]][link[1]] -= 1.0

            for link in self.static_network.link_list:
                if new_capacity_matrix[link[0]][link[1]] < 1:
                    blocked_links.append(link)

            new_blocked_path.append(blocked_links)

        for t in range(len(self.disturbed_links)):
            if t < len(new_blocked_path):
                new_blocked_path[t].extend(self.disturbed_links[t])
            else:
                new_blocked_path.append(self.disturbed_links[t])

        self.blocked_path = new_blocked_path

    def update_path_usage(self):
        if len(self.path_usage) > 0:
            self.path_usage.pop(0)

    def check_conflict(self, path):
        conflict = False
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) in self.path_usage[i]:
                conflict = True
            if conflict:
                break
        return conflict

    def create_time_expanded_network_all(self, waiting_cost=None, wait_in_advance=False):
        if waiting_cost is None:  # if not specified then apply global waiting cost
            waiting_cost = self.waiting_cost

        node_num = self.static_network.node_num
        t_max = len(self.blocked_path)
        nodes = self.static_network.nodes

        expanded_node_num = node_num * (t_max + 1)
        expanded_adjacency_matrix = np.zeros((expanded_node_num, expanded_node_num))
        adjacency_matrix = self.static_network.adjacency_matrix

        if wait_in_advance:
            wait_coef = 0.0
        else:
            wait_coef = EPSILON / (t_max + 10)

        for t in range(t_max):
            # occupied_paths = self.path_usage[t]
            occupied_paths = self.blocked_path[t]
            for i in range(len(nodes)):
                expanded_adjacency_matrix[t * node_num + i][
                    (t + 1) * node_num + i] = waiting_cost - wait_coef * t + EPSILON  # > 0
                for j in nodes[i].neighbors:
                    if not (i, j) in occupied_paths:
                        expanded_adjacency_matrix[t * node_num + i][(t + 1) * node_num + j] = adjacency_matrix[i][j]

        for i in range(len(nodes)):
            for j in nodes[i].neighbors:
                expanded_adjacency_matrix[t_max * node_num + i][t_max * node_num + j] = adjacency_matrix[i][j]

        return expanded_adjacency_matrix

    def create_time_expanded_network(self, origin, waiting_cost=None, wait_in_advance=False):
        # Unlike create_time_expanded_network_all,
        # this function wires links that are only reachable from the origin at each time step
        if waiting_cost is None:  # if not specified then apply global waiting cost
            waiting_cost = self.waiting_cost

        node_num = self.static_network.node_num
        t_max = len(self.blocked_path)
        nodes = self.static_network.nodes

        expanded_node_num = node_num * (t_max + 1)
        expanded_adjacency_matrix = np.zeros((expanded_node_num, expanded_node_num))
        adjacency_matrix = self.static_network.adjacency_matrix

        if wait_in_advance:
            wait_coef = 0.0
        else:
            wait_coef = EPSILON / (t_max + 10)

        reachable_nodes = [origin]
        for t in range(t_max):
            newly_reached_nodes = []
            # occupied_paths = self.path_usage[t]
            occupied_paths = self.blocked_path[t]
            for i in range(len(nodes)):
                if i in reachable_nodes:
                    expanded_adjacency_matrix[t * node_num + i][
                        (t + 1) * node_num + i] = waiting_cost - wait_coef * t + EPSILON  # > 0
                    for j in nodes[i].neighbors:
                        if not (i, j) in occupied_paths:
                            expanded_adjacency_matrix[t * node_num + i][(t + 1) * node_num + j] = adjacency_matrix[i][j]
                            newly_reached_nodes.append(j)

            reachable_nodes.extend(newly_reached_nodes)

        for i in range(len(nodes)):
            for j in nodes[i].neighbors:
                expanded_adjacency_matrix[t_max * node_num + i][t_max * node_num + j] = adjacency_matrix[i][j]

        return expanded_adjacency_matrix

    def create_time_expanded_network_fixed_path(self, fixed_path, waiting_cost=None, wait_in_advance=False):
        if waiting_cost is None:  # if not specified then apply global waiting cost
            waiting_cost = self.waiting_cost

        node_num = self.static_network.node_num
        t_max = len(self.blocked_path)
        nodes = self.static_network.nodes

        expanded_node_num = node_num * (t_max + 1)
        expanded_adjacency_matrix = np.zeros((expanded_node_num, expanded_node_num))
        adjacency_matrix = self.static_network.adjacency_matrix
        fixed_paths_tuple = []

        if wait_in_advance:
            wait_coef = 0.0
        else:
            wait_coef = EPSILON / (t_max + 10)

        for t in range(len(fixed_path) - 1):
            fixed_paths_tuple.append((fixed_path[t], fixed_path[t + 1]))

        for t in range(t_max):
            occupied_paths = self.blocked_path[t]

            for i in range(len(nodes)):
                expanded_adjacency_matrix[t * node_num + i][
                    (t + 1) * node_num + i] = waiting_cost - wait_coef * t + EPSILON
                for j in nodes[i].neighbors:
                    if (i, j) in fixed_paths_tuple:
                        if not (i, j) in occupied_paths:
                            expanded_adjacency_matrix[t * node_num + i][(t + 1) * node_num + j] = adjacency_matrix[i][j]

        for i in range(len(nodes)):
            for j in nodes[i].neighbors:
                if (i, j) in fixed_paths_tuple:
                    expanded_adjacency_matrix[t_max * node_num + i][t_max * node_num + j] = adjacency_matrix[i][j]

        return expanded_adjacency_matrix

    def get_and_register_adaptive_fastest_path(self, origin, destination, waiting_cost=None):
        # fastest path considering availability of links
        expanded_adjacency_matrix = csr_matrix(self.create_time_expanded_network(origin, waiting_cost=waiting_cost))
        distance, path_plan = self.find_fastest_path(expanded_adjacency_matrix, origin, destination)
        self.add_path_plan(path_plan)
        return distance, path_plan

    def get_and_register_wait_fastest_path(self, origin, destination, waiting_cost=None, fixed_path=None):
        # fastest path based on a given path. if not provided shortest path is set
        if fixed_path is None:
            free_shortest_path = self.static_network.nodes[origin].shortest_paths[destination]
        else:
            free_shortest_path = fixed_path

        expanded_adjacency_matrix = csr_matrix(self.create_time_expanded_network_fixed_path(free_shortest_path,
                                                                                            waiting_cost=waiting_cost))
        distance, path_plan = self.find_fastest_path(expanded_adjacency_matrix, origin, destination)
        self.add_path_plan(path_plan)
        return distance, path_plan

    def compare_paths(self, origin, destination, waiting_cost=None, fixed_path=None):
        if fixed_path is None:
            free_shortest_path = self.static_network.nodes[origin].shortest_paths[destination]
        else:
            free_shortest_path = fixed_path

        expanded_adjacency_matrix_wait = csr_matrix(self.create_time_expanded_network_fixed_path(free_shortest_path,
                                                                                                 waiting_cost=waiting_cost))

        expanded_adjacency_matrix_adaptive = csr_matrix(self.create_time_expanded_network(waiting_cost=waiting_cost))

        _, path_original_wait = self.find_fastest_path(expanded_adjacency_matrix_wait)
        _, path_original_adaptive = self.find_fastest_path(expanded_adjacency_matrix_adaptive)

        if len(path_original_adaptive) > len(path_original_wait):
            print("--------------------")
            print(path_original_wait)
            print(path_original_adaptive)
            print("--------------------")

    def find_fastest_path(self, expanded_adjacency_matrix, origin, destination):
        distance, last_visited = shortest_path(expanded_adjacency_matrix,
                                               indices=origin,
                                               directed=True, return_predecessors=True)

        t_max = len(self.blocked_path)
        node_num = self.static_network.node_num

        for t_scan in range(t_max + 1):
            if not np.isinf(distance[t_scan * node_num + destination]):
                current_node = t_scan * node_num + destination
                path = []
                for k in range((t_scan + 1) * node_num):
                    path.append(current_node)
                    current_node = last_visited[current_node]
                    if current_node == origin:  # fin
                        path.append(current_node)

                        path = path[::-1]  # (list(reversed(path)))
                        break
                path_original = self.__convert_location_expand_to_original(node_num, path)
                return distance[t_scan * node_num + destination], path_original

        raise Exception("No shortest path found!")

    def __convert_location_expand_to_original(self, node_num, path_expanded):
        path_original = []
        for i in path_expanded:
            path_original.append(i % node_num)

        return path_original
