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


def compute_fisher_diag(model, dataloader, max_batches=0, estimator="sample"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]

    num_samples = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        use_non_blocking = torch.cuda.is_available()
        data = data.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)
        num_samples += labels.size(0)

        # Calculate output log probabilities
        log_probs = torch.nn.functional.log_softmax(model(data), dim=1)
        selected_log_probs = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)

        if estimator == "batch":
            grad1 = autograd.grad(selected_log_probs.mean(), model.parameters(), create_graph=False)
            for fisher_diag_value, grad_value in zip(fisher_diag, grad1):
                fisher_diag_value.add_(grad_value.detach() ** 2 * labels.size(0))
            del grad1
            continue

        for i, log_prob in enumerate(selected_log_probs):

            # Calculate first-order derivatives (gradients)
            retain_graph = i < labels.size(0) - 1
            grad1 = autograd.grad(log_prob, model.parameters(), create_graph=False, retain_graph=retain_graph)

            # Update Fisher diagonal elements
            for fisher_diag_value, grad_value in zip(fisher_diag, grad1):
                fisher_diag_value.add_(grad_value.detach() ** 2)
                
            # Free up memory by removing computation graph
            del log_prob, grad1

        # Release CUDA memory
        # torch.cuda.empty_cache()

    if num_samples == 0:
        raise ValueError("Cannot compute Fisher information from an empty dataloader")

    # Calculate the mean value
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
