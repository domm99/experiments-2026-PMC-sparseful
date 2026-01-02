import os
import sys
import yaml
import torch
import random
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from learning.model import MLP
from learning import prune_model, initialize_model
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


def get_current_device():
    device: str = 'cpu'
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            device = current_accelerator.type
    return device


def run_simulation(threshold,
                   sparsity_level,
                   number_subregions,
                   seed,
                   pre_pruning = False,
                   pruning_for_check = False,
                   dataset='EMNIST',
                   device = 'cpu'):

    simulator = Simulator()

    # deformed lattice
    simulator.environment.set_neighborhood_function(radius_neighborhood(1.15))
    deformed_lattice(simulator, 7, 7, 1, 0.01)

    initial_model_params = initialize_model(dataset).state_dict()
    if pre_pruning:
        initial_model_params = prune_model(initial_model_params, sparsity_level, dataset).state_dict() # Pre pruning

    devices = len(simulator.environment.nodes.values())
    mapping_devices_area = distribute_nodes_spatially(devices, number_subregions)

    print(f'Number of devices: {devices}')
    print(mapping_devices_area)

    train_data, test_data = download_dataset(dataset)

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
            seed = seed,
            pruning_for_check=pruning_for_check,
            device = device,
            dataset_name=dataset,)
    # render
    # simulator.schedule_event(0.95, render_sync, simulator, "result")
    config = ExporterConfig('data/', f'federations_seed-{seed}_regions-{number_subregions}_sparsity-{sparsity_level}', [], [], 3)
    simulator.schedule_event(0.96, federations_count_csv_exporter, simulator, 1.0, config)
    config = ExporterConfig('data/', f'experiment_seed-{seed}_regions-{number_subregions}_sparsity-{sparsity_level}', ['TrainLoss', 'ValidationLoss', 'ValidationAccuracy'], ['mean', 'std', 'min', 'max'], 3)
    simulator.schedule_event(1.0, csv_exporter, simulator, 1.0, config)
    simulator.add_monitor(TestSetEvalMonitor(simulator))
    simulator.run(80)

if __name__ == '__main__':

    # Hyper-parameters from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seed', default='1')
    parser.add_argument('--dataset', default='EMNIST')
    args = parser.parse_args()

    # Hyper-parameters configuration
    thresholds = [40.0]
    sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    areas = [3, 5, 9]
    seeds = range(int(args.max_seed))
    device = get_current_device()
    dataset = args.dataset

    experiment_log_dir = 'finished-experiments/'

    data_dir = Path(experiment_log_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_file = f'{experiment_log_dir}experiment_log.csv'

    df = pd.DataFrame(columns=['timestamp', 'experiment'])

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        pass

    print(f'-------------------- USING {device} --------------------')

    for seed in seeds:
        random.seed(seed)
        for sparsity_level in sparsity_levels:
            for threshold in thresholds:
                for a in areas:
                    print(f'Starting simulation with seed={seed}, regions={a}, sparsity={sparsity_level}, threshold={threshold}')
                    run_simulation(threshold, sparsity_level, a, seed, pre_pruning = True, pruning_for_check = False, dataset=dataset, device=device)
                    experiment_name = f'seed-{seed}_regions-{a}_sparsity-{sparsity_level}_threshold-{threshold}'
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    new_line = {'timestamp': timestamp, 'experiment': experiment_name}
                    df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
                    df.to_csv(csv_file, index=False)