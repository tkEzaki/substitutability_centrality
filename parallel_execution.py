import simulate_optimal_transport
import pandas as pd
from multiprocessing import Pool
import time
import uuid
import pickle
import os
from itertools import product


pd.set_option('display.max_columns', 50)

DEFAULT_PARAM = {
    "num_packets": 1, "waiting_cost": 0, "packet_type": "TFP",
    "dyn_dem_change_params": None, "one_shot_dem_change_params": None,
    "T_max": 1000, "condition_memo": "memo", "monitor": False
}

TESTED_PARAM_LIST = {
    "num_packets": [1, 20, 40, 60, 80, 100], "waiting_cost": [0], "packet_type": ["TFP"],
    "dyn_dem_change_params": [None], "one_shot_dem_change_params": [None],
    "T_max": [1000], "condition_memo": ["memo"], "monitor": [False]
}


class ParallelExecutor():
    def __init__(self, folder_name="test"):
        self.param_list = TESTED_PARAM_LIST.copy()
        self.network_list = None
        self.folder_name = folder_name

    def execute(self, core_num=None):
        if self.network_list is None:
            raise Exception("No network list is set!")

        self.generate_condition_list()
        if core_num is None:
            core_num = os.cpu_count() - 2

        print(len(self.condition_list), "conditions executed with", core_num, "cores.")
        start = time.time()
        with Pool(core_num) as pool:
            pool.map(self.simulator_wrapper, self.condition_list)

        print("Execution finished.", time.time() - start, "[sec]")

    def generate_condition_list(self):
        self.condition_list = []
        if len(self.network_list) == 0:
            raise Exception("Network list is empty.")
        keys, values = zip(*self.param_list.items())
        param_list = [dict(zip(keys, p)) for p in product(*values)]
        for network in self.network_list:
            for params in param_list:
                self.condition_list.append((network, params))

        return

    def simulator_wrapper(self, condition_data):
        network, kwargs = condition_data[0], condition_data[1]

        results = simulate_optimal_transport.simulate(network, **kwargs)
        file_name = self.folder_name + "/" + str(uuid.uuid4()) + ".pkl"

        with open(file_name, "wb") as tf:
            pickle.dump(results, tf)
