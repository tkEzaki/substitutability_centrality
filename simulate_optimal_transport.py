import components
import numpy as np
import random
import time
import datetime
from operator import attrgetter

np.set_printoptions(threshold=np.inf)


class DynamicalDemandChangeParams():
    def __init__(self, block_period: int, block_num: int, id_sort=False, reverse=False, blocked_link_candidates=None) -> None:
        self.block_period = block_period
        self.block_num = block_num
        self.id_sort = id_sort
        self.reverse = reverse
        self.blocked_link_candidates = blocked_link_candidates
        if blocked_link_candidates is not None:
            if len(blocked_link_candidates) < block_num:
                raise Exception("too large block num is set!")


class OneShotDemandChangeParams():
    def __init__(self, packet_size: int, genrate_pattern=None) -> None:
        self.packet_size = packet_size
        self.generate_pattern = genrate_pattern


class OptimalTransportSimulator():
    def __init__(self, static_network: components.Network, waiting_cost):
        self.static_network = static_network
        self.path_planner = components.PathPlanner(static_network, waiting_cost)
        self.packets = []
        self.next_packet_id = 0
        self.waiting_cost = waiting_cost
        self.env_is_changed = False

        # results
        self.cost_records = []
        self.move_records = []
        self.path_length_records = []
        self.num_packets_records = []
        self.recording = False
        self.monitor = False
        self.disturbed_links_plan = []
        self.traced_steps_records = []
        self.traced_cost_records = []

    def generate_origin_and_destination(self, generate_pattern=None):
        if generate_pattern is None and self.static_network.network_name in ["japan_grid", "japan_hub_and_spoke"]:
            origin = random.choices(list(range(self.static_network.node_num)), weights=self.static_network.origin_demand)[0]
            destination = random.choices(list(range(self.static_network.node_num)), weights=self.static_network.destination_demand[origin])[0]
        elif generate_pattern is None:  # if no generate pattern is specified, then use that defined in network instance.
            generate_pattern = self.static_network.generate_pattern
            if generate_pattern is None:  # if no generate pattern is defined on the network instance
                origin = random.randrange(self.static_network.node_num)
                destination = random.randrange(self.static_network.node_num)
                while destination == origin:
                    destination = random.randrange(self.static_network.node_num)
            else:  # use generate pattern defined on the network instance
                origin = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[0])[0]
                destination = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[1])[0]
                while destination == origin:
                    destination = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[1])[0]
        else:  # use specified generate pattern
            origin = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[0])[0]
            destination = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[1])[0]
            while destination == origin:
                destination = random.choices(list(range(self.static_network.node_num)), weights=generate_pattern[1])[0]

        return origin, destination

    def add_single_TFP_packet(self, origin, destination, steps=0, total_cost=0, is_traced=False, trace_id=-1):
        distance, path_plan = self.path_planner.get_and_register_adaptive_fastest_path(origin, destination,
                                                                                       waiting_cost=self.waiting_cost)

        packet = components.Packet("TFP", origin, destination, self.next_packet_id, path_plan=path_plan, distance=distance,
                                   steps=steps, total_cost=total_cost, is_traced=is_traced, trace_id=trace_id)
        self.next_packet_id += 1
        self.packets.append(packet)
        if self.monitor:
            print("Packet created. OD:", (origin, destination), ", path:", path_plan)

    def add_single_TFP_packet_random(self, generate_pattern=None, is_traced=False, packet_size=1, trace_id=-1):
        origin, destination = self.generate_origin_and_destination(generate_pattern=generate_pattern)

        for i in range(packet_size):
            self.add_single_TFP_packet(origin, destination, is_traced=is_traced, trace_id=trace_id)

    def add_single_AFP_packet(self, origin, destination, steps=0, total_cost=0, is_traced=False, trace_id=-1):
        distance, path_plan = self.path_planner.get_and_register_adaptive_fastest_path(origin, destination,
                                                                                       waiting_cost=self.waiting_cost)

        packet = components.Packet("AFP", origin, destination, self.next_packet_id, path_plan=path_plan, distance=distance,
                                   steps=steps, total_cost=total_cost, is_traced=is_traced, trace_id=trace_id)
        self.next_packet_id += 1
        self.packets.append(packet)
        if self.monitor:
            print("Packet created. OD:", (origin, destination), ", path:", path_plan)

    def add_single_AFP_packet_random(self, generate_pattern=None, is_traced=False, packet_size=1, trace_id=-1):
        origin, destination = self.generate_origin_and_destination(generate_pattern=generate_pattern)

        for i in range(packet_size):
            self.add_single_AFP_packet(origin, destination, is_traced=is_traced, trace_id=trace_id)

    def add_single_SSP_packet(self, origin, destination, steps=0, total_cost=0, is_traced=False, trace_id=-1):
        distance, path_plan = self.path_planner.get_and_register_wait_fastest_path(
            origin, destination, waiting_cost=self.waiting_cost
        )

        packet = components.Packet(
            "SSP", origin, destination, self.next_packet_id, path_plan=path_plan,
            distance=distance, steps=steps, total_cost=total_cost, is_traced=is_traced, trace_id=trace_id
        )
        self.next_packet_id += 1
        self.packets.append(packet)
        if self.monitor:
            print("Packet created. OD:", (origin, destination), ", path:", path_plan)

    def add_single_SSP_packet_random(self, generate_pattern=None, packet_size=1, is_traced=False, trace_id=-1):
        origin, destination = self.generate_origin_and_destination(generate_pattern=generate_pattern)

        for i in range(packet_size):
            self.add_single_SSP_packet(origin, destination, is_traced=is_traced, trace_id=trace_id)

    def add_single_packet_wait_path(self, packet_type, fixed_path, steps=0, total_cost=0, is_traced=False, trace_id=-1):
        distance, path_plan = self.path_planner.get_and_register_wait_fastest_path(
            fixed_path[0], fixed_path[-1], waiting_cost=self.waiting_cost, fixed_path=fixed_path
        )

        packet = components.Packet(packet_type, fixed_path[0], fixed_path[-1], self.next_packet_id, path_plan=path_plan,
                                   distance=distance, steps=steps, total_cost=total_cost,
                                   is_traced=is_traced, trace_id=trace_id
                                   )
        self.next_packet_id += 1
        self.packets.append(packet)
        if self.monitor:
            print("Packet created. OD:", (fixed_path[0], fixed_path[-1]), ", path:", path_plan)

    def move_and_regenerate_packets(self, generate_pattern=None):
        arrived_packets = []
        moved_packets = []
        num_arrived_traced_packets = 0
        arrived_packets_type_list = []

        for i in range(len(self.packets)):
            self.packets[i].steps += 1
            self.static_network.node_usage_record[self.packets[i].current_location] += 1

            if not self.packets[i].current_location == self.packets[i].path_plan[1]:  # moved
                moved_packets.append(i)
                self.packets[i].total_cost += \
                    self.static_network.adjacency_matrix[self.packets[i].path_plan[0]][self.packets[i].path_plan[1]]
                self.static_network.link_usage_record[self.packets[i].path_plan[0], self.packets[i].path_plan[1]] += 1
            else:
                self.packets[i].total_cost += self.waiting_cost

            self.packets[i].path_plan.pop(0)
            self.packets[i].current_location = self.packets[i].path_plan[0]
            self.packets[i].update_remaining_path_length()
            self.packets[i].shortest_remaining_path_length = \
                len(self.static_network
                    .nodes[self.packets[i].current_location]
                    .shortest_paths[self.packets[i].destination])

            if len(self.packets[i].path_plan) == 1:  # arrived
                arrived_packets.append(i)
                if not self.packets[i].is_traced:
                    arrived_packets_type_list.append(self.packets[i].packet_type)

                if self.recording:
                    if self.packets[i].is_traced:
                        num_arrived_traced_packets += 1
                        if len(self.traced_steps_records) <= self.packets[i].trace_id:
                            self.traced_steps_records.append([])
                            self.traced_cost_records.append([])
                            self.traced_steps_records[self.packets[i].trace_id].append(self.packets[i].steps)
                            self.traced_cost_records[self.packets[i].trace_id].append(self.packets[i].total_cost)
                        else:
                            self.traced_steps_records[self.packets[i].trace_id].append(self.packets[i].steps)
                            self.traced_cost_records[self.packets[i].trace_id].append(self.packets[i].total_cost)
                    else:
                        self.path_length_records.append(self.packets[i].steps)
                        self.cost_records.append(self.packets[i].total_cost)

        if len(self.path_planner.blocked_path) > 0:
            self.path_planner.blocked_path.pop(0)

        if len(self.disturbed_links_plan) > 0:
            self.disturbed_links_plan.pop(0)
            self.path_planner.disturbed_links.pop(0)

        num_deleted = 0
        for i in arrived_packets:
            del self.packets[i - num_deleted]
            num_deleted += 1

        if self.recording:
            self.move_records.append(len(moved_packets))
            self.num_packets_records.append(len(self.packets) + len(arrived_packets))

        self.path_planner.update_path_usage()

        if self.monitor:
            print("Packets moved.", len(self.packets), "packets. Replenish new packets.---move_and_regenerate_packets")

        for packet_type in arrived_packets_type_list:
            {"TFP": self.add_single_TFP_packet_random,
             "SSP": self.add_single_SSP_packet_random,
             "AFP": self.add_single_AFP_packet_random}[packet_type](generate_pattern=generate_pattern)

        if self.monitor:
            print("Replenishment done.", len(self.packets), "packets---move_and_regenerate_packets")

        return len(arrived_packets), len(moved_packets)

    def generate_traced_packets(self, packet_size, packet_type, trace_id, generate_pattern=None):

        {"TFP": self.add_single_TFP_packet_random,
         "SSP": self.add_single_SSP_packet_random,
         "AFP": self.add_single_AFP_packet_random}[packet_type](
            generate_pattern=generate_pattern, packet_size=packet_size, is_traced=True, trace_id=trace_id
        )

    def regenerate_all_paths(self, reverse=False, id_sort=False, distance_sort=False):
        if distance_sort:
            self.packets.sort(key=attrgetter("shortest_remaining_path_length"), reverse=reverse)

        if id_sort:
            self.packets.sort(key=attrgetter("packet_id"), reverse=reverse)

        self.path_planner = components.PathPlanner(self.static_network, waiting_cost=self.path_planner.waiting_cost)  # new
        self.path_planner.blocked_path = self.disturbed_links_plan.copy()
        self.path_planner.disturbed_links = self.disturbed_links_plan.copy()

        packets_temp = self.packets.copy()
        self.packets = []
        self.next_packet_id -= len(packets_temp)

        if self.monitor:
            print("Re-optimize path.---regenerate_all_paths")

        for packet in packets_temp:
            origin = packet.current_location
            destination = packet.destination

            {"TFP": self.add_single_TFP_packet,
             "SSP": self.add_single_SSP_packet,
             "AFP": self.add_single_AFP_packet}[packet.packet_type](
                origin, destination, steps=packet.steps, total_cost=packet.total_cost,
                is_traced=packet.is_traced, trace_id=packet.trace_id
            )

        if self.monitor:
            print("Re-optimization done.---regenerate_all_paths")

    def disturbance_update_all_paths(self, dyn_dem_change_params: DynamicalDemandChangeParams):
        disturbed_links = []
        if dyn_dem_change_params.blocked_link_candidates is None:
            for i in range(dyn_dem_change_params.block_num):
                disturbed_node = random.randrange(self.static_network.node_num)
                next_node = random.choice(self.static_network.nodes[disturbed_node].neighbors)
                while (disturbed_node, next_node) in disturbed_links:
                    disturbed_node = random.randrange(self.static_network.node_num)
                    next_node = random.choice(self.static_network.nodes[disturbed_node].neighbors)
                disturbed_links.append((disturbed_node, next_node))
        else:
            link_num_list = [ln for ln in range(len(dyn_dem_change_params.blocked_link_candidates))]
            selected_links = random.sample(link_num_list, dyn_dem_change_params.block_num)
            for link_num in selected_links:
                disturbed_links.append(dyn_dem_change_params.blocked_link_candidates[link_num])

        disturbed_links_plan = [disturbed_links.copy() for i in range(dyn_dem_change_params.block_period)]
        if self.monitor:
            print("Disturbed_link_updated:", disturbed_links_plan, "---disturbance_update_all_paths")

        # sort before copy
        if dyn_dem_change_params.id_sort:
            self.packets.sort(key=attrgetter("packet_id"), reverse=dyn_dem_change_params.reverse)
        else:
            self.packets.sort(key=attrgetter("shortest_remaining_path_length"), reverse=dyn_dem_change_params.reverse)

        # recreate
        self.path_planner = components.PathPlanner(self.static_network, self.path_planner.waiting_cost)  # new
        packets_temp = self.packets.copy()
        self.packets = []
        self.next_packet_id -= len(packets_temp)
        self.path_planner.blocked_path = disturbed_links_plan.copy()
        self.path_planner.disturbed_links = disturbed_links_plan.copy()

        self.disturbed_links_plan = disturbed_links_plan.copy()

        if self.monitor:
            print("Re-calculate path plans.---disturbance_update_all_paths")

        for packet in packets_temp:
            if packet.packet_type == "TFP" or packet.packet_type == "SSP":
                self.add_single_packet_wait_path(
                    packet.packet_type,
                    packet.path_plan, steps=packet.steps, total_cost=packet.total_cost,
                    is_traced=packet.is_traced, trace_id=packet.trace_id
                )
            elif packet.packet_type == "AFP":
                origin = packet.current_location
                destination = packet.destination
                self.add_single_AFP_packet(
                    origin, destination, steps=packet.steps, total_cost=packet.total_cost,
                    is_traced=packet.is_traced, trace_id=packet.trace_id
                )

        if self.monitor:
            print("Path plans recalculated.---disturbance_update_all_paths")

    def compute_average_cost(self, threshold=0):
        return np.mean(self.cost_records[threshold:])

    def compute_std_cost(self, threshold=0):
        return np.std(self.cost_records[threshold:])

    def compute_average_move(self, threshold=0):
        return np.mean(self.move_records[threshold:])

    def compute_std_move(self, threshold=0):
        return np.std(self.move_records[threshold:])

    def compute_average_path_length(self, threshold=0):
        return np.mean(self.path_length_records[threshold:])

    def compute_std_path_length(self, threshold=0):
        return np.std(self.path_length_records[threshold:])

    def compute_average_num_packets(self, threshold=0):
        return np.mean(self.num_packets_records[threshold:])

    def compute_average_steps_traced_packets(self):
        if len(self.traced_steps_records) > 1:
            steps_ndarray = np.array(self.traced_steps_records[:-1])
            return steps_ndarray.mean(axis=0)
        else:
            return [0]

    def compute_std_steps_traced_packets(self):
        if len(self.traced_steps_records) > 1:
            steps_ndarray = np.array(self.traced_steps_records[:-1])
            return steps_ndarray.std(axis=0)
        else:
            return [0]

    def compute_average_cost_traced_packets(self):
        if len(self.traced_cost_records) > 1:
            cost_ndarray = np.array(self.traced_cost_records[:-1])
            return cost_ndarray.mean(axis=0)
        else:
            return [0]

    def compute_std_cost_traced_packets(self):
        if len(self.traced_cost_records) > 1:
            cost_ndarray = np.array(self.traced_cost_records[:-1])
            return cost_ndarray.std(axis=0)
        else:
            return [0]


def simulate(
    network, num_packets=1, waiting_cost=0, packet_type="TFP",
    dyn_dem_change_params=None, one_shot_dem_change_params=None,
    T_max=1000, condition_memo="memo", monitor=False
):
    time_start = time.time()
    ots = OptimalTransportSimulator(network, waiting_cost)
    ots.recording = True
    ots.monitor = monitor

    for i in range(num_packets):  # initialize particles
        {"TFP": ots.add_single_TFP_packet_random,
         "SSP": ots.add_single_SSP_packet_random,
         "AFP": ots.add_single_AFP_packet_random}[packet_type]()

    if one_shot_dem_change_params is not None:
        current_trace_id = 0
        ots.generate_traced_packets(
            one_shot_dem_change_params.packet_size, packet_type, trace_id=current_trace_id,
            generate_pattern=one_shot_dem_change_params.generate_pattern
        )

    # set disruption
    if dyn_dem_change_params is not None:
        ots.disturbance_update_all_paths(
            dyn_dem_change_params
        )

    for t in range(T_max):
        ots.env_is_changed = False
        ots.move_and_regenerate_packets()

        if one_shot_dem_change_params is not None:
            if len(ots.packets) == num_packets:  # no additional traced packets
                current_trace_id += 1
                ots.generate_traced_packets(
                    one_shot_dem_change_params.packet_size, packet_type, trace_id=current_trace_id,
                    generate_pattern=one_shot_dem_change_params.generate_pattern
                )

        if dyn_dem_change_params is not None:
            if t % dyn_dem_change_params.block_period == 0:  # update disturbance
                ots.disturbance_update_all_paths(dyn_dem_change_params)
                ots.env_is_changed = True

        # if packet_type == "AFP":
        if packet_type == "AFP" and ots.env_is_changed:
            ots.regenerate_all_paths()

    return {
        "execution_info": {
            "condition_memo": condition_memo,
            "execution_time": time.time() - time_start,
            "date_of_execution": str(datetime.datetime.today()),
        },
        "simulation_condition": {
            "num_packets": num_packets,
            "waiting_cost": waiting_cost,
            "packet_type": packet_type,
            "dyn_dem_change_block_num": None if dyn_dem_change_params is None else dyn_dem_change_params.block_num,
            "dyn_dem_change_block_period": None if dyn_dem_change_params is None else dyn_dem_change_params.block_period,
            "generate_pattern": network.generate_pattern,
            "traced_packet_size": None if one_shot_dem_change_params is None else one_shot_dem_change_params.packet_size,
            "generate_pattern_trace": None if one_shot_dem_change_params is None else one_shot_dem_change_params.generate_pattern,
            "T_max": T_max,
            "capacity_coef": ots.static_network.capacity_coef,
            "capacity_type": ots.static_network.capacity_type,
            "mixture_ratio": ots.static_network.mixture_ratio,
            "network_name": ots.static_network.network_name,
        },
        "results": {
            "average_num_packets": ots.compute_average_num_packets(),
            "average_move": ots.compute_average_move(),
            "std_move": ots.compute_std_move(),
            "average_path_length": ots.compute_average_path_length(),
            "std_path_length": ots.compute_std_path_length(),
            "average_cost": ots.compute_average_cost(),
            "std_cost": ots.compute_std_cost(),
            "average_path_length_traced": ots.compute_average_steps_traced_packets(),
            "std_path_length_traced": ots.compute_std_steps_traced_packets(),
            "average_cost_traced": ots.compute_average_cost_traced_packets(),
            "std_cost_traced": ots.compute_std_cost_traced_packets(),
            "node_usage": ots.static_network.node_usage_record / T_max,
            "link_usage": ots.static_network.link_usage_record / T_max,
        },
        "network_data": ots.static_network,
    }


if __name__ == '__main__':
    results = []
    num_packets_list = [50]
    start = time.time()
    network = components.ConnectedHubAndSpokeNetwork(5)

    for num_packets in num_packets_list:
        print(simulate(network, num_packets=150, T_max=1000))

    print(time.time() - start)
