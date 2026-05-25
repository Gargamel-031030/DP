import torch
from options import parse_args
from torch import autograd
from moments_accountant import MomentsAccountant
import math

args = parse_args()


def compute_noise_multiplier(N, L, T, epsilon, delta):
    q = (1.0 * L) / N
    nm = 10 * q * math.sqrt(T * (-math.log10(delta))) / epsilon
    return nm

def compute_noise_multiplier_decay(target_epsilon, target_delta, global_epoch, local_steps, L, N, decay_factor):
    init_sigma = 10.0
    last_sigma = init_sigma
    q = (1.0 * L) / N
    flag = True
    while flag:
        accountant = MomentsAccountant(epsilon=target_epsilon, delta=target_delta, noise_multiplier=init_sigma)
        eps = 0.0
        for i in range(int(global_epoch)):
            eps = accountant.get_privacy_spent(sigma=init_sigma * (decay_factor ** i), q=q, steps=local_steps,
                                               target_delta=target_delta)
            # print(eps)
        # if (eps * 8) < target_epsilon:
        #     last_sigma = init_sigma
        #     init_sigma -= 0.01
        if eps < target_epsilon:
            last_sigma = init_sigma
            init_sigma -= 0.01
        else:
            flag = False
    return last_sigma


def compute_fisher_diag(model, dataloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        # Calculate output log probabilities
        log_probs = torch.nn.functional.log_softmax(model(data), dim=1)

        for i, label in enumerate(labels):
            log_prob = log_probs[i, label]

            # Calculate first-order derivatives (gradients)
            model.zero_grad()
            grad1 = autograd.grad(log_prob, model.parameters(), create_graph=True, retain_graph=True)

            # Update Fisher diagonal elements
            for fisher_diag_value, grad_value in zip(fisher_diag, grad1):
                fisher_diag_value.add_(grad_value.detach() ** 2)
                
            # Free up memory by removing computation graph
            del log_prob, grad1

        # Release CUDA memory
        # torch.cuda.empty_cache()

    # Calculate the mean value
    num_samples = len(dataloader.dataset)
    fisher_diag = [fisher_diag_value / num_samples for fisher_diag_value in fisher_diag]

    # Normalize Fisher values layer-wise
    normalized_fisher_diag = []
    for fisher_value in fisher_diag:
        x_min = torch.min(fisher_value)
        x_max = torch.max(fisher_value)
        denom = x_max - x_min
        if denom.item() <= 0 or not torch.isfinite(denom).item():
            normalized_fisher_value = torch.zeros_like(fisher_value)
        else:
            normalized_fisher_value = (fisher_value - x_min) / denom
        normalized_fisher_diag.append(normalized_fisher_value)

    return normalized_fisher_diag
