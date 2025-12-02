from learning.model import MLP
from phyelds.data import Field
from phyelds.libraries.collect import collect_with
from phyelds.libraries.device import local_id, store
from phyelds.calculus import aggregate, neighbors, remember
from phyelds.libraries.leader_election import elect_leaders
from phyelds.libraries.spreading import distance_to, broadcast
from learning import local_training, model_evaluation, average_weights, post_prune_model


impulsesEvery = 5
MAX_TIME = 80

@aggregate
def psfl_client(initial_model_params, data, threshold, sparsity_level, regions, seed):

    hyperparams = f'seed-{seed}_regions-{regions}_sparsity-{sparsity_level}'
    training_data, validation_data, test_data = data
    stored_model = remember((initial_model_params, 0)) # Stores local model and current global round
    local_model_weights, tick = stored_model.value
    local_model = load_from_weights(local_model_weights)

    # labels_train = [training_data[idx][1] for idx in range(len(training_data))]
    # labels_val = [validation_data[idx][1] for idx in range(len(validation_data))]
    # labels_test = [test_data[idx][1] for idx in range(len(test_data))]
    # print(f'Client {local_id()} --> training labels {set(labels_train)} -- validation labels {set(labels_val)} -- testing labels {set(labels_test)}')

    # Local training
    evolved_model, train_loss = local_training(local_model, 2, training_data, 128)
    validation_accuracy, validation_loss = model_evaluation(evolved_model, validation_data, 128)

    pruned_model = post_prune_model(evolved_model, sparsity_level)
    log(train_loss, validation_loss, validation_accuracy) # Metrics logging

    distances = loss_based_distances(pruned_model, validation_data)
    leader = elect_leaders(threshold, distances) # If leader is true, then I'm an aggregator
    store('is_aggregator', leader)
    potential = distance_to(leader, distances)

    models = collect_with(potential, [evolved_model], lambda x, y: x + y)
    aggregated_model = average_weights(models, [1.0 for _ in models])
    area_model = broadcast(leader, aggregated_model, distances)

    if tick % impulsesEvery == 0:
        avg = average_weights([evolved_model, area_model], [0.1, 0.9])
        stored_model.update((avg, tick + 1))
    else:
        stored_model.update((evolved_model, tick + 1))

    if tick == MAX_TIME:
        store('final_model', evolved_model)
        store('test_data', test_data)
        store('hyperparams', hyperparams)

    return potential


@aggregate
def loss_based_distances(model_weights, validation_data):
    models_weights = neighbors(model_weights)
    neighbors_models = Field(models_weights.exclude_self(), local_id())
    evaluations = neighbors_models.map(lambda m: model_evaluation(m, validation_data, 128)[1])
    neighbors_evaluations = neighbors(evaluations.data)
    loss_field = compute_loss_metric(evaluations, neighbors_evaluations.data)
    return loss_field


@aggregate
def compute_loss_metric(evaluations, neighbors_evaluations):
    mid = evaluations.node_id
    data = evaluations.data
    loss_metric = dict()
    for neighbor_id, evaluation in data.items():
        neighbor_evaluation_of_myself = neighbors_evaluations[neighbor_id].get(mid, float('inf'))
        loss_metric[neighbor_id] = neighbor_evaluation_of_myself + evaluation
    return Field(loss_metric, mid)


def log(train_loss, validation_loss, validation_accuracy):
    store('TrainLoss', train_loss)
    store('ValidationLoss', validation_loss)
    store('ValidationAccuracy', validation_accuracy)


def load_from_weights(weights):
    model = MLP()
    model.load_state_dict(weights)
    return model