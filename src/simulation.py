import os
import sys
import yaml
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
from learning.model import MLP
from PSFLClient import psfl_client
from dummy_client import dummy_client
from phyelds.simulator import Simulator
from utils import distribute_nodes_spatially
from phyelds.simulator.render import render_sync
from TestSetEvalMonitor import TestSetEvalMonitor
from phyelds.simulator.deployments import deformed_lattice
from phyelds.simulator.runner import aggregate_program_runner
from custom_exporter import federations_count_csv_exporter
from phyelds.simulator.neighborhood import radius_neighborhood
from phyelds.simulator.exporter import csv_exporter, ExporterConfig
from ProFed.partitioner import Environment, Region, download_dataset, split_train_validation, partition_to_subregions


def get_hyperparameters():
    """
    Fetches the hyperparameters from the docker compose config file
    :return: the experiment name and the hyperparameters (as a dictionary name -> values)
    """
    hyperparams = os.environ['LEARNING_HYPERPARAMETERS']
    hyperparams = yaml.safe_load(hyperparams)
    experiment_name, hyperparams = list(hyperparams.items())[0]
    return experiment_name.lower(), hyperparams

def run_simulation(threshold, sparsity_level, number_subregions, seed):

    simulator = Simulator()

    # deformed lattice
    simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
    deformed_lattice(simulator, 7, 7, 1, 0.01)

    initial_model_params = MLP().state_dict()

    devices = len(simulator.environment.nodes.values())
    mapping_devices_area = distribute_nodes_spatially(devices, number_subregions)

    print(f'Number of devices: {devices}')
    print(mapping_devices_area)

    train_data, test_data = download_dataset('EMNIST')

    train_data, validation_data = split_train_validation(train_data, 0.8)
    print(f'Number of training samples: {len(train_data)}')
    environment = partition_to_subregions(train_data, validation_data, 'Hard', number_subregions, seed)
    test_data, _ = split_train_validation(test_data, 1.0)
    environment_test = partition_to_subregions(test_data, test_data, 'Hard', number_subregions, seed)

    mapping = {}

    for region_id, devices in mapping_devices_area.items():
        mapping_devices_data = environment.from_subregion_to_devices(region_id, len(devices))
        mapping_devices_data_test = environment_test.from_subregion_to_devices(region_id, len(devices))
        for device_index, data in mapping_devices_data.items():
            device_id = devices[device_index]
            test_subset, _ = mapping_devices_data_test[device_index]
            complete_data = data[0], data[1], test_subset
            mapping[device_id] = complete_data

    # schedule the main function
    for node in simulator.environment.nodes.values():
        simulator.schedule_event(
            0.0,
            aggregate_program_runner,
            simulator,
            1.0,
            node,
            psfl_client,
            data=mapping[node.id],
            initial_model_params=initial_model_params,
            threshold=threshold,
            sparsity_level=sparsity_level,
            regions=number_subregions,
            seed = seed)
        # simulator.schedule_event(
        #         0.0,
        #         aggregate_program_runner,
        #         simulator,
        #         0.1,
        #         node,
        #         dummy_client,
        #         data=mapping[node.id]
        # )
    # render
    # simulator.schedule_event(0.95, render_sync, simulator, "result")
    config = ExporterConfig('data/', f'federations_seed-{seed}_regions-{number_subregions}_sparsity-{sparsity_level}', [], [], 3)
    simulator.schedule_event(0.96, federations_count_csv_exporter, simulator, 1.0, config)
    config = ExporterConfig('data/', f'experiment_seed-{seed}_regions-{number_subregions}_sparsity-{sparsity_level}', ['TrainLoss', 'ValidationLoss', 'ValidationAccuracy'], ['mean', 'std', 'min', 'max'], 3)
    simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.add_monitor(TestSetEvalMonitor(simulator))
    simulator.run(80)

# Hyper-parameters configuration
thresholds = [20.0]
sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.9]
# areas = [3, 5, 9]
seeds = list(range(10))

experiment_name, hyperparams = get_hyperparameters()
areas = hyperparams['areas']

experiment_log_dir = 'finished-experiments/'

data_dir = Path(experiment_log_dir)
data_dir.mkdir(parents=True, exist_ok=True)

csv_file = f'{experiment_log_dir}experiment_log.csv'

df = pd.DataFrame(columns=['timestamp', 'experiment'])

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    pass

for seed in seeds:
    random.seed(seed)
    for a in areas:
        for threshold in thresholds:
            for sparsity_level in sparsity_levels:
                print(f'Starting simulation with seed={seed}, regions={a}, sparsity={sparsity_level}, threshold={threshold}')
                run_simulation(threshold, sparsity_level, a, seed)
                experiment_name = f'seed-{seed}_regions-{a}_sparsity-{sparsity_level}_threshold-{threshold}'
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_line = {'timestamp': timestamp, 'experiment': experiment_name}
                df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
                df.to_csv(csv_file, index=False)
