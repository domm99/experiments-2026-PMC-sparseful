import pandas as pd
from phyelds.simulator import Monitor
from learning import model_evaluation

class TestSetEvalMonitor(Monitor):

    def __init__(self, simulator):
        super().__init__(simulator)

    def on_finish(self) -> None:
        nodes = list(self.simulator.environment.nodes.values())
        experiments_hyperparams = nodes[0].data['outputs']['hyperparams']
        accuracies = []

        for node in nodes:
            model_weights = node.data['outputs']['final_model']
            test_data = node.data['outputs']['test_data']
            accuracy, loss = model_evaluation(model_weights, test_data, 64)
            accuracies.append(accuracy)

        df = pd.DataFrame({ 'Accuracy': [sum(accuracies)/len(accuracies)] })
        df.to_csv(f'data/test_{experiments_hyperparams}.csv', index=False)
