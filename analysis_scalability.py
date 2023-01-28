from components import *
import time

if __name__ == "__main__":

    coef_list = [1, 2, 5]
    for coef in coef_list:
        for i in range(10):
            lattice_network = GeneralHubAndSpokeNetwork(4 * coef, 5 * coef)
            time_start = time.perf_counter()
            lattice_network.set_substitutability_centrality()
            time_end = time.perf_counter()
            print(coef, time_end - time_start)
