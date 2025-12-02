import copy
import torch
from torch import nn
from learning.model import MLP
import torch.nn.utils.prune as tprune
from torch.utils.data import DataLoader


def local_training(model, epochs, data, batch_size):
    # torch.manual_seed(seed)
    criterion = nn.NLLLoss()
    model.train()
    epoch_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        batch_loss = []
        for batch_index, (images, labels) in enumerate(data_loader):
            model.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        mean_epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(mean_epoch_loss)
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def model_evaluation(model_params, data, batch_size):
    model = MLP()
    model.load_state_dict(model_params)
    criterion = nn.NLLLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for batch_index, (images, labels) in enumerate(data_loader):
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


def average_weights(models_params, weights):
    w_avg = copy.deepcopy(models_params[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], 0.0)
    sum_weights = sum(weights)
    for key in w_avg.keys():
        for i in range(0, len(models_params)):
            w_avg[key] += models_params[i][key] * weights[i]
        w_avg[key] = torch.div(w_avg[key], sum_weights)
    return w_avg

def post_prune_model(model_params, amount):
    model = MLP()
    model.load_state_dict(model_params)
    # Pruning
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.l1_unstructured(module, name='weight', amount=amount)

    #Remove the pruning reparametrizations to make the model explicitly sparse
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.remove(module, 'weight')
    return model.state_dict()
