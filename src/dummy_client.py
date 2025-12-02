from learning.model import MLP
from phyelds.data import Field
from phyelds.libraries.collect import collect_with
from phyelds.libraries.device import local_id, store
from phyelds.calculus import aggregate, neighbors, remember
from phyelds.libraries.leader_election import elect_leaders
from phyelds.libraries.spreading import distance_to, broadcast
from learning import local_training, model_evaluation, average_weights


impulsesEvery = 5


@aggregate
def dummy_client(data):
    training_data, validation_data = data
    training_labels =  set([training_data[idx][1] for idx in range(len(training_data))])
    validation_labels = set([validation_data[idx][1] for idx in range(len(validation_data))])
    print(f'Device {local_id()} TRAINING -> {training_labels}')
    print(f'Device {local_id()} VALIDATION -> {validation_labels}')
    return local_id()
