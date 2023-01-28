from copy import deepcopy
import components
import simulate_optimal_transport
import parallel_execution


def example_execution():
    pe = parallel_execution.ParallelExecutor("results")
    network_list = []
    # generate 10 networks with random cost values
    for _ in range(10):
        chsns = components.ConnectedHubAndSpokeNetwork(5)
        chsns.set_generate_pattern_distorted(w=3)
        chsns.set_shortest_path_and_link_bw_centrality()
        chsns.set_substitutability_centrality()

        for capacity_coef in [20]:
            for mr in [0, 0.3]:
                network = deepcopy(chsns)
                network.set_capacity_dist_substitutability(capacity_coef=capacity_coef, mixture_ratio=mr, consider_demand=True)
                network_list.append(network)

    pe.network_list = network_list

    # set lists of parameter conditions
    pe.param_list["num_packets"] = [10, 200, 400, 600, 800, 1000]
    pe.param_list["packet_type"] = ["AFP"]
    pe.param_list["dyn_dem_change_params"] = [
        None,
        simulate_optimal_transport.DynamicalDemandChangeParams(5, 3, blocked_link_candidates=chsns.core_links),
        simulate_optimal_transport.DynamicalDemandChangeParams(5, 6, blocked_link_candidates=chsns.core_links)
    ]

    # run simulations
    pe.execute(core_num=1)


if __name__ == "__main__":
    example_execution()
