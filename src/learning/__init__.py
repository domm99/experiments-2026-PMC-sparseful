import copy
import torch
from torch import nn
from learning.model import MLP
import torch.nn.utils.prune as tprune
from torch.utils.data import DataLoader

def local_training(model, epochs, data, batch_size, device):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        batch_loss = []
        for batch_index, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        mean_epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(mean_epoch_loss)
    return model.cpu().state_dict(), sum(epoch_loss) / len(epoch_loss)


def model_evaluation(model_params, data, batch_size, device):
    model = MLP()
    model.load_state_dict(model_params)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
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


def prune_model(model_params, amount):
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


def check_sparsity(state_dict, verbose=False):
    total_zeros = 0
    total_params = 0

    for name, tensor in state_dict.items():

        num_params = tensor.numel()
        num_zeros = torch.sum(tensor == 0).item()

        total_params += num_params
        total_zeros += num_zeros

        if verbose:
            layer_sparsity = (num_zeros / num_params) * 100
            print(f"Layer: {name} | Sparsity: {layer_sparsity:.2f}%")

    if total_params == 0:
        return 0.0

    global_sparsity = (total_zeros / total_params) * 100
    return global_sparsity