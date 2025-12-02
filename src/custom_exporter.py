import os
import pandas as pd
from pathlib import Path
from dataclasses import replace
from phyelds.simulator import Simulator
from phyelds.simulator.exporter import ExporterConfig


def federations_count_csv_exporter(simulator: Simulator, time_delta: float, config: ExporterConfig, **kwargs):
    file_path = f'{config.output_directory}{config.experiment_name}.csv'
    if not os.path.exists(file_path) or config.initial:
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        df = init_dataframe(config, file_path)
    else:
        df = pd.read_csv(file_path)
    nodes = simulator.environment.nodes.values()
    nodes = [node for node in nodes if node.data['outputs']['is_aggregator']]
    count = len(nodes)
    df = pd.concat([df, pd.DataFrame([{'FederationsCount': count}])], ignore_index=True)
    df.to_csv(file_path, mode='w', index=False)
    config = replace(config, initial=False)
    simulator.schedule_event(
        time_delta, federations_count_csv_exporter, simulator, time_delta, config, **kwargs
    )


def init_dataframe(config, file_path):
    """Initializes an empty DataFrame with appropriate columns based on config."""
    columns = ['FederationsCount']
    try:
        os.remove(file_path)
    except OSError:
        pass
    return pd.DataFrame(columns=columns).astype('float64')