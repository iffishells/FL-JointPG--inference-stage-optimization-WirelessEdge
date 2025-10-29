# splitgp_cifar10.py
# VGG11 implementation adapted for CIFAR-10 with model-splitting utilities

import argparse
import os
import time
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Add a guarded tqdm import so script doesn't fail if tqdm isn't installed
try:
    from tqdm import tqdm
except Exception:
    # fallback: identity function
    def tqdm(iterable, *args, **kwargs):
        return iterable


class AuxiliaryClassifier(nn.Module):
    """A small classifier that takes the client-side feature map and predicts CIFAR-10 classes.
    It applies adaptive avgpool then a linear layer. Pool size is configurable to match paper's auxiliary classifier size.
    """

    def __init__(self, in_channels: int, num_classes: int = 10, pool_size: Tuple[int, int] = (1, 1)):
        super().__init__()
        self.pool_size = pool_size
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flattened = in_channels * pool_size[0] * pool_size[1]
        self.fc = nn.Linear(flattened, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def make_vgg11_bn_for_cifar(num_classes: int = 10) -> torchvision.models.VGG:
    """Return a VGG-11-BN model adapted for CIFAR (32x32 input).
    We use torchvision's vgg11_bn and adapt the classifier for CIFAR-10.
    """
    # Use pretrained=False to keep it lightweight and reproducible
    vgg = torchvision.models.vgg11_bn(pretrained=False)
    # Adjust classifier: original VGG expects 7x7 feature maps; CIFAR has smaller spatial dims
    # Replace classifier with a small MLP suitable for CIFAR-10
    vgg.classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return vgg


def split_vgg11(vgg: torchvision.models.VGG, cut_at: Optional[int] = None) -> Tuple[nn.Module, nn.Module, int]:
    """Split VGG-11 (with bn) into client-side (features[:cut_at]) and server-side (features[cut_at:]+classifier).

    Args:
      vgg: a torchvision VGG model (vgg.features and vgg.classifier expected)
      cut_at: index in vgg.features (0..len(features)). If None, choose a conservative split near 10% of params.

    Returns:
      client_module, server_module, client_out_channels

    The client_module outputs the feature map (tensor) that becomes server_module input.
    """
    features = list(vgg.features)
    total_layers = len(features)

    if cut_at is None:
        # Heuristic: choose cut such that client params ~10% of total
        # We'll approximate by layer counts; choose after first conv block (after ReLU/MaxPool sequence)
        # For vgg11_bn, reasonable split index is 6 (after first two conv blocks)
        cut_at = 6

    if not (0 < cut_at < total_layers):
        raise ValueError(f"cut_at must be between 1 and {total_layers-1}")

    client_features = nn.Sequential(*features[:cut_at])
    server_features = nn.Sequential(*features[cut_at:])

    # Build server module which expects the output of client_features and continues forward
    class ServerModule(nn.Module):
        def __init__(self, server_feats: nn.Sequential, classifier: nn.Sequential):
            super().__init__()
            self.server_feats = server_feats
            # adapt classifier: we assume after server_feats we can apply adaptive pool -> classifier
            # we'll replace classifier head to accept adaptive pooled 512-dim inputs
            # take classifier but if its first Linear expects 512*7*7, we adapt by replacing it
            # We'll create a simple head matching our make_vgg11_bn_for_cifar design
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            # If classifier ends with Linear(512, num_classes) as in our factory, reuse it
            # Otherwise, create a new linear using last layer output features
            if isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
                out_lin = classifier[-1]
                in_features = out_lin.in_features
                out_features = out_lin.out_features
                # If in_features != 512, we will still use out_features but map from 512
                if in_features != 512:
                    self.head = nn.Linear(512, out_features)
                else:
                    # reuse the head weights shape
                    self.head = nn.Linear(in_features, out_features)
            else:
                self.head = nn.Linear(512, 10)

        def forward(self, x):
            x = self.server_feats(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x

    server_module = ServerModule(server_features, vgg.classifier)

    # Compute client out channels by passing a dummy tensor
    with torch.no_grad():
        device = next(vgg.parameters()).device if any(p.requires_grad for p in vgg.parameters()) else torch.device('cpu')
        dummy = torch.zeros(1, 3, 32, 32).to(device)
        client_out = client_features(dummy)
        client_out_channels = client_out.shape[1]

    return client_features, server_module, client_out_channels


def get_dataloaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def train_epoch(client_model: nn.Module, aux_classifier: nn.Module, server_model: nn.Module,
                dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device, gamma: float = 0.5):
    client_model.train()
    aux_classifier.train()
    server_model.train()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    # iterate with tqdm so users can see progress per-epoch
    for inputs, targets in tqdm(dataloader, desc='Train', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward on client
        features = client_model(inputs)

        # Client-side prediction via auxiliary classifier
        aux_logits = aux_classifier(features)

        # Server-side prediction via server model (simulate offloading)
        server_logits = server_model(features.detach())

        loss_client = criterion(aux_logits, targets)
        loss_server = criterion(server_logits, targets)

        loss = gamma * loss_client + (1.0 - gamma) * loss_server

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = aux_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(client_model: nn.Module, aux_classifier: nn.Module, server_model: nn.Module,
             dataloader: DataLoader, device: torch.device, eth: Optional[float] = None):
    client_model.eval()
    aux_classifier.eval()
    server_model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    import math
    # progress bar for evaluation
    # Counters for offloading decisions
    total_sent_to_server = 0
    total_handled_on_client = 0

    for inputs, targets in tqdm(dataloader, desc='Eval', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        features = client_model(inputs)
        aux_logits = aux_classifier(features)

        batch_size = inputs.size(0)
        # If eth is None: use full model (server) for all
        if eth is None:
            logits = server_model(features)
            sent_to_server = batch_size
            handled_on_client = 0
        else:
            # compute entropy per-sample from aux_logits softmax
            probs = F.softmax(aux_logits, dim=1)
            # Use a small epsilon to avoid log(0)
            eps = 1e-12
            entropy = -(probs * (probs + eps).log()).sum(dim=1)  # Shannon-like
            to_server = entropy > eth
            sent_to_server = int(to_server.sum().item())
            handled_on_client = batch_size - sent_to_server
            logits = aux_logits.clone()
            if to_server.any():
                srv_inp = features[to_server]
                srv_out = server_model(srv_inp)
                logits[to_server] = srv_out

        # accumulate counts
        total_sent_to_server += sent_to_server
        total_handled_on_client += handled_on_client

        loss = criterion(logits, targets)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0

    # Print a concise summary of offloading decisions for this evaluation call
    print(f"Eval summary: total_samples={total}, "
          f"sent_to_server={total_sent_to_server}, "
          f"handled_on_client={total_handled_on_client} ,"
          f"sent_ratio={total_sent_to_server/total:.4f}, "
          f"handled_ratio={total_handled_on_client/total:.4f} ,"
          f"eth={eth if eth is not None else 'N/A'}")

    # expose summary stats on the function object for callers that want to programmatically access them
    try:
        evaluate.last_stats = {
            'total_samples': int(total),
            'sent_to_server': int(total_sent_to_server),
            'handled_on_client': int(total_handled_on_client),
            'eth':float(eth) if eth is not None else None
        }
    except Exception:
        # in case evaluate is not writable for some reason, silently ignore
        pass

    return avg_loss, acc


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# Added federated split training utilities and client partitioning to match the paper's setup
import copy
import random
from collections import defaultdict
from torch.utils.data import Subset


def partition_dataset_noniid(dataset, num_shards=100, shards_per_client=2, num_clients=50, seed=0):
    """
    Partition dataset into `num_shards` shards sorted by label and assign each client `shards_per_client` shards.
    Returns a list of indices for each client.
    This mirrors the paper's non-IID shard allocation.
    """
    # Build mapping label -> list of indices
    label_to_idx = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_idx[label].append(idx)

    # Create list of (label, idx) flattened then sort by label (already grouped)
    all_indices = []
    for label in sorted(label_to_idx.keys()):
        all_indices.extend(label_to_idx[label])

    total_samples = len(all_indices)
    assert total_samples >= num_shards, "num_shards too large"
    shards = []
    shard_size = total_samples // num_shards
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else total_samples
        shards.append(all_indices[start:end])

    # shuffle shards and assign to clients
    random.seed(seed)
    shard_ids = list(range(num_shards))
    random.shuffle(shard_ids)

    client_indices = []
    for k in range(num_clients):
        assigned = []
        for s in range(shards_per_client):
            sid = shard_ids[(k * shards_per_client + s) % num_shards]
            assigned.extend(shards[sid])
        client_indices.append(assigned)

    return client_indices


def build_client_dataloaders(trainset, client_indices, batch_size=50, num_workers=2):
    client_loaders = []
    for idxs in client_indices:
        subset = Subset(trainset, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        client_loaders.append(loader)
    return client_loaders


def build_client_testsets(testset, client_indices, trainset, rho=0.0, seed=0):
    """Construct per-client test subsets: all main class samples plus rho portion of OOD samples (random).
    rho = #OOD / #main as defined in the paper.
    Returns list of index lists for each client (relative to testset).
    """
    # First map class -> indices in testset
    label_to_idx = defaultdict(list)
    for idx, (_, label) in enumerate(testset):
        label_to_idx[label].append(idx)

    client_test_indices = []
    random.seed(seed)
    for train_idxs in client_indices:
        # determine main classes present in the client's train indices
        main_labels = set()
        for tidx in train_idxs:
            _, lab = trainset[tidx]
            main_labels.add(lab)
        # gather all test indices with those labels
        main_test_idxs = []
        for lab in main_labels:
            main_test_idxs.extend(label_to_idx[lab])
        num_main = len(main_test_idxs)
        num_ood = int(round(rho * num_main))
        # gather OOD candidate indices
        ood_candidates = [i for lab, idxs in label_to_idx.items() if lab not in main_labels for i in idxs]
        ood_selected = random.sample(ood_candidates, min(num_ood, len(ood_candidates))) if num_ood > 0 else []
        client_test_indices.append(main_test_idxs + ood_selected)
    return client_test_indices


# Replace the centralized train loop with federated split training
# We'll add a function `federated_train` that implements Algorithm 1 from the paper

def federated_train(vgg_factory_fn,
                    num_clients: int = 50,
                    num_shards: int = 100,
                    shards_per_client: int = 2,
                    global_rounds: int = 800,
                    local_epochs: int = 1,
                    batch_size: int = 50,
                    lr: float = 0.01,
                    gamma: float = 0.5,
                    lam: float = 0.2,
                    device: torch.device = torch.device('cpu'),
                    save_path: Optional[str] = None,
                    rho: float = 0.0,
                    eth: Optional[float] = None,
                    num_workers: int = 2,
                    seed: int = 0):
    """
    Federated split training simulation that follows the SplitGP steps in the paper.
    Returns per-client models (phi_k, kappa_k) and global server model theta.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Prepare datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    client_indices = partition_dataset_noniid(trainset, num_shards=num_shards, shards_per_client=shards_per_client,
                                             num_clients=num_clients, seed=seed)
    client_loaders = build_client_dataloaders(trainset, client_indices, batch_size=batch_size, num_workers=num_workers)
    client_test_indices = build_client_testsets(testset, client_indices, trainset, rho=rho, seed=seed)

    # Build initial global model and split
    global_vgg = vgg_factory_fn()
    # choose cut_at=13 to match the paper's reported split (close to their |phi|/|theta| sizes)
    client_template, server_template, client_out_ch = split_vgg11(global_vgg, cut_at=13)
    # Use pool_size=(2,2) so auxiliary classifier input dim = client_out_ch * 4 = 1024 (256*2*2)
    # This yields aux params ~10,250 as reported in the paper.
    aux_template = AuxiliaryClassifier(client_out_ch, num_classes=10, pool_size=(2, 2))

    # Initialize global server (theta) and client templates
    global_theta = copy.deepcopy(server_template)  # this is the server-side global model
    # initialize per-client phi and kappa as copies of template
    client_phis = [copy.deepcopy(client_template).to(device) for _ in range(num_clients)]
    client_kappas = [copy.deepcopy(aux_template).to(device) for _ in range(num_clients)]
    # Send global theta to device
    global_theta = global_theta.to(device)

    # Count params for info
    # Paper reports |phi| including the auxiliary classifier and |theta| excluding it.
    phi_including_kappa = count_parameters(client_phis[0]) + count_parameters(client_kappas[0])
    theta_excluding_kappa = count_parameters(global_theta) - count_parameters(client_kappas[0])
    print(f"Initial params (paper-style): |phi|={phi_including_kappa:,}, |theta|={theta_excluding_kappa:,}, |kappa|={count_parameters(client_kappas[0]):,}")

    # Per-client dataset sizes for alpha weighting
    client_sizes = [len(idxs) for idxs in client_indices]
    total_size = sum(client_sizes)
    alphas = [s / total_size for s in client_sizes]

    # Training loop across global rounds
    best_val = 0.0
    # Add tqdm for outer rounds so user sees progress across global rounds
    for rnd in tqdm(range(global_rounds), desc='Global rounds'):
        # Store updated thetas and phis from clients
        updated_thetas_state = []
        updated_phis_state = []
        updated_kappas_state = []

        # For each client perform local training using current global theta
        for k in tqdm(range(num_clients), desc=f'Clients (rnd {{rnd+1}})', leave=False):
            client_phi = client_phis[k]
            client_kappa = client_kappas[k]
            # local copy of global theta
            local_theta = copy.deepcopy(global_theta).to(device)

            # build optimizer over client's phi, kappa, and local theta parameters
            optim_params = list(client_phi.parameters()) + list(client_kappa.parameters()) + list(local_theta.parameters())
            optimizer = optim.SGD(optim_params, lr=lr, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()

            # local epochs (paper uses 1 epoch per global round)
            loader = client_loaders[k]
            for le in range(local_epochs):
                # show progress across minibatches for this client's local training
                for inputs, targets in tqdm(loader, desc=f'Client {k} local epoch {le+1}', leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    features = client_phi(inputs)
                    aux_logits = client_kappa(features)
                    # server-side logits computed with local_theta (no detach to allow gradient flow to phi)
                    # Use a clone of features to avoid in-place modification issues from ReLU(inplace=True) inside the server module
                    server_logits = local_theta(features.clone())
                    loss_c = criterion(aux_logits, targets)
                    loss_s = criterion(server_logits, targets)
                    loss = gamma * loss_c + (1.0 - gamma) * loss_s
                    loss.backward()
                    optimizer.step()

            # collect updated states
            updated_thetas_state.append({k: v.cpu() for k, v in local_theta.state_dict().items()})
            updated_phis_state.append({k: v.cpu() for k, v in client_phi.state_dict().items()})
            updated_kappas_state.append({k: v.cpu() for k, v in client_kappa.state_dict().items()})

        # Aggregate server-side model theta: weighted average across clients
        new_theta_state = copy.deepcopy(updated_thetas_state[0])
        for key in new_theta_state.keys():
            new_theta_state[key] = sum(alphas[i] * updated_thetas_state[i][key] for i in range(num_clients))
        global_theta.load_state_dict(new_theta_state)

        # Aggregate client-side models: compute average phi and kappa then apply local aggregation phi_k <- lam*phi_k + (1-lam)*phi_avg
        # compute phi_avg
        phi_avg_state = copy.deepcopy(updated_phis_state[0])
        for key in phi_avg_state.keys():
            phi_avg_state[key] = sum(alphas[i] * updated_phis_state[i][key] for i in range(num_clients))
        kappa_avg_state = copy.deepcopy(updated_kappas_state[0])
        for key in kappa_avg_state.keys():
            kappa_avg_state[key] = sum(alphas[i] * updated_kappas_state[i][key] for i in range(num_clients))

        # apply aggregation to each client phi/ kappa
        for k in range(num_clients):
            # load updated phi into client_phis[k]
            client_phis[k].load_state_dict(updated_phis_state[k])
            client_kappas[k].load_state_dict(updated_kappas_state[k])
            # perform phi_k <- lam * phi_k + (1-lam) * phi_avg
            cur = client_phis[k].state_dict()
            for key in cur.keys():
                cur[key] = lam * cur[key].cpu() + (1.0 - lam) * phi_avg_state[key]
            client_phis[k].load_state_dict({k: v.to(device) for k, v in cur.items()})
            # kappa aggregation
            curk = client_kappas[k].state_dict()
            for key in curk.keys():
                curk[key] = lam * curk[key].cpu() + (1.0 - lam) * kappa_avg_state[key]
            client_kappas[k].load_state_dict({k: v.to(device) for k, v in curk.items()})

        # Optionally evaluate after certain rounds; we'll evaluate every 10 rounds to save time
        if (rnd + 1) % max(1, global_rounds // 10) == 0 or rnd == global_rounds - 1:
            # Evaluate average client accuracy using per-client test subsets and selective offloading with eth
            global_theta.eval()
            accs = []
            for k in tqdm(range(num_clients), desc='Eval clients', leave=False):
                client_test_idxs = client_test_indices[k]
                if len(client_test_idxs) == 0:
                    continue
                subset = Subset(testset, client_test_idxs)
                loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=0)
                val_loss, val_acc = evaluate(client_phis[k].to(device), client_kappas[k].to(device), global_theta.to(device), loader, device, eth=eth)
                accs.append(val_acc)
            avg_acc = sum(accs) / len(accs) if accs else 0.0
            print(f"Round {rnd+1}/{global_rounds}  AvgClientValAcc {avg_acc:.2f}%")
            if avg_acc > best_val:
                best_val = avg_acc
                if save_path:
                    torch.save({
                        'global_theta': global_theta.state_dict(),
                        'client_phis': [c.state_dict() for c in client_phis],
                        'client_kappas': [k.state_dict() for k in client_kappas],
                        'round': rnd,
                        'avg_acc': avg_acc,
                    }, save_path)

    return client_phis, client_kappas, global_theta


# Now update main() to expose federated training options and defaults from the paper
# ...existing code...

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib


def evaluate_clients_models(client_phis, client_kappas, global_theta, testset, client_indices_for_rhos, eth=None, device=torch.device('cpu')):
    """Evaluate models for each client and return average accuracy across clients."""
    global_theta = global_theta.to(device)
    global_theta.eval()
    accs = []
    for k, test_indices in enumerate(client_indices_for_rhos):
        if not test_indices:
            continue
        subset = Subset(testset, test_indices)
        loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=0)
        val_loss, val_acc = evaluate(client_phis[k].to(device), client_kappas[k].to(device), global_theta.to(device), loader, device, eth=eth)
        accs.append(val_acc)
    avg_acc = sum(accs) / len(accs) if accs else 0.0
    return avg_acc


def sweep_and_save_results(client_phis, client_kappas, global_theta, trainset, testset,
                           num_clients, shards, shards_per_client, rho_list, eth_list, device,
                           model_name: str = 'vgg11', dataset_name: str = 'cifar10',
                           method_name: str = 'splitgp', global_rounds: int = 0, lam: float = 0.2,
                           out_base: str = 'results'):
    """
    Perform sweep over rho_list and eth_list and save results into a hierarchical results folder:
      results/{model_name}/{dataset_name}/{method_name}/rounds_{global_rounds}/lambda_{lam}/
    Produces: results.csv (combined), results.png (plot), and per-eth CSV files under eth_results/.
    """
    # Create output directory hierarchy
    out_dir = pathlib.Path(out_base) / model_name / dataset_name / method_name / f'rounds_{global_rounds}' / f'lambda_{lam}'
    out_dir.mkdir(parents=True, exist_ok=True)
    eth_dir = out_dir / 'eth_results'
    eth_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / 'results.csv'
    out_png = out_dir / 'results.png'

    # We'll produce CSV columns: rho, eth, acc_client_only, acc_server_only, acc_selective
    header = ['rho', 'eth', 'acc_client_only', 'acc_server_only', 'acc_selective']
    rows = []

    # Precompute client_indices (train partition) once so main classes per client are consistent across rhos
    client_indices = partition_dataset_noniid(trainset, num_shards=shards, shards_per_client=shards_per_client, num_clients=num_clients, seed=0)

    for rho in rho_list:
        # build test subsets for this rho
        client_test_indices = build_client_testsets(testset, client_indices, trainset, rho=rho, seed=0)

        # client-only evaluation: eth very large so no offload -> set eth = +inf to force client predict
        acc_client_only = evaluate_clients_models(client_phis, client_kappas, global_theta, testset, client_test_indices, eth=1e9, device=device)
        # server-only evaluation: use server predictions only
        acc_server_only = evaluate_clients_models_server_only(client_phis, client_kappas, global_theta, testset, client_test_indices, device=device)

        # For faster incremental saving, also collect per-eth rows
        for eth in eth_list:
            acc_selective = evaluate_clients_models(client_phis, client_kappas, global_theta, testset, client_test_indices, eth=eth, device=device)
            row = {'rho': rho, 'eth': eth, 'acc_client_only': acc_client_only, 'acc_server_only': acc_server_only, 'acc_selective': acc_selective}
            rows.append(row)

        # Save per-rho/eth intermediate results if you like (we'll save per-eth CSVs after loop)

    # Write combined CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Also write per-eth CSVs
    for eth in eth_list:
        eth_file = eth_dir / f'eth_{eth}.csv'
        eth_rows = [r for r in rows if float(r['eth']) == float(eth)]
        with open(eth_file, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for rr in eth_rows:
                w.writerow(rr)

    # Plot: for each eth, plot acc_selective vs rho; also plot client_only and server_only as baselines
    plt.figure(figsize=(10, 6))
    rhos = sorted(list(set([r['rho'] for r in rows])))
    # compute base baselines (same for all eth rows) as arrays
    bas_client = {rho: next((r['acc_client_only'] for r in rows if r['rho'] == rho), None) for rho in rhos}
    bas_server = {rho: next((r['acc_server_only'] for r in rows if r['rho'] == rho), None) for rho in rhos}

    # plot baselines
    plt.plot(rhos, [bas_client[r] for r in rhos], '--', color='gray', label='client-only')
    plt.plot(rhos, [bas_server[r] for r in rhos], ':', color='black', label='server-only')

    # for each eth plot selective
    for eth in eth_list:
        selective_vals = [next(r['acc_selective'] for r in rows if r['rho'] == rho and float(r['eth']) == float(eth)) for rho in rhos]
        plt.plot(rhos, selective_vals, marker='o', label=f'eth={eth}')

    plt.xlabel('rho (p)')
    plt.ylabel('Test accuracy (%)')
    plt.title(f'{model_name.upper()} {dataset_name.upper()} {method_name.upper()}: accuracy vs rho for different Eth (client/server baselines)')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png)
    print(f'Wrote CSV to {out_csv} and plot to {out_png}')


def evaluate_clients_models_server_only(client_phis, client_kappas, global_theta, testset, client_indices_for_rhos, device=torch.device('cpu')):
    # server-only evaluation: feed client features through phi but always use server output
    global_theta = global_theta.to(device)
    global_theta.eval()
    accs = []
    for k, test_indices in enumerate(client_indices_for_rhos):
        if not test_indices:
            continue
        subset = Subset(testset, test_indices)
        loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=0)
        # evaluate but force server-only: we can call evaluate with eth=None which uses server for all
        val_loss, val_acc = evaluate(client_phis[k].to(device), client_kappas[k].to(device), global_theta.to(device), loader, device, eth=None)
        accs.append(val_acc)
    avg_acc = sum(accs) / len(accs) if accs else 0.0
    return avg_acc


def main():
    parser = argparse.ArgumentParser(description='VGG11 SplitGP federated split training for CIFAR-10 (paper reproduction)')
    parser.add_argument('--global-rounds', type=int, default=800, help='Number of global rounds (paper uses 800 for CIFAR-10)')
    parser.add_argument('--clients', type=int, default=50, help='Number of clients K (default: 50)')
    parser.add_argument('--shards', type=int, default=100, help='Number of shards (default:100)')
    parser.add_argument('--shards-per-client', type=int, default=2, help='Number of shards per client (default:2)')
    parser.add_argument('--local-epochs', type=int, default=1, help='Local epochs per global round (paper uses 1)')
    parser.add_argument('--batch-size', type=int, default=50, help='Local minibatch size (paper uses 50)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (paper uses 0.01)')
    parser.add_argument('--gamma', type=float, default=0.5, help='Weight for client-side loss (gamma)')
    parser.add_argument('--lambda', dest='lam', type=float, default=0.2, help='Lambda for client aggregation (paper uses 0.2)')
    parser.add_argument('--eth', type=float, default=0.2, help='Entropy threshold for selective offload during eval (paper explores multiple values)')
    parser.add_argument('--rho', type=float, default=0.0, help='Relative portion of OOD test samples (rho)')
    parser.add_argument('--rho-list', type=str, default='0.0,0.2,0.4,0.6,0.8,1.0', help='Comma-separated list of rho values to sweep')
    parser.add_argument('--eth-list', type=str, default='0.05,0.1,0.2,0.4,0.8,1.2,1.6,2.3', help='Comma-separated list of eth values to sweep')
    parser.add_argument('--eval-only', action='store_true', help='Skip training and run evaluation sweep on saved checkpoint')
    parser.add_argument('--save', type=str, default='vgg_split_federated.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--quick', action='store_true', help='Quick mode (small clients/shards) for smoke tests')
    # Auto-save helpers: generate descriptive checkpoint filenames for easier tracking
    parser.add_argument('--auto-save', action='store_true', help='Auto-generate a descriptive checkpoint filename and save there')
    parser.add_argument('--tag', type=str, default='', help='Optional short tag to include in auto-generated checkpoint filename')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to store auto-generated checkpoints')
    args = parser.parse_args()

    # Auto-generate descriptive checkpoint filename if requested
    if getattr(args, 'auto_save', False):
        ts = time.strftime('%Y%m%d-%H%M%S')
        tag = args.tag.strip() if getattr(args, 'tag', None) else ''
        def _clean(x):
            return str(x).replace('/', '_').replace(' ', '_')
        model_str = 'vgg11'
        fname = f"round_{_clean(args.global_rounds)}_lambda_{_clean(args.lam)}_gamm_{_clean(args.gamma)}_{model_str}"
        if tag:
            fname = f"{_clean(tag)}_{fname}"
        fname = f"{fname}_{ts}.pth"
        save_dir = getattr(args, 'save_dir', 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        auto_path = os.path.join(save_dir, fname)
        args.save = auto_path
        print(f"Auto-generated checkpoint path: {args.save}")

    device = torch.device(args.device)

    # If quick mode is on, reduce clients/shards for smoke testing
    if args.quick:
        args.clients = min(4, args.clients)
        args.shards = min(8, args.shards)
        args.shards_per_client = min(2, args.shards_per_client)
        args.global_rounds = min(1, args.global_rounds)

    print('Starting federated split training with settings:')
    print(vars(args))

    # If eval-only, load checkpoint and run sweep
    if args.eval_only:
        if not os.path.exists(args.save):
            raise FileNotFoundError(f'Checkpoint {args.save} not found')
        ckpt = torch.load(args.save, map_location=device)
        # load global theta and client models
        global_theta = make_vgg11_bn_for_cifar()
        # build template split to construct server module
        client_template, server_template, client_out_ch = split_vgg11(global_theta, cut_at=13)
        global_theta = server_template.to(device)
        global_theta.load_state_dict(ckpt['global_theta'])
        client_phis = [copy.deepcopy(client_template).to(device) for _ in range(args.clients)]
        client_kappas = [copy.deepcopy(AuxiliaryClassifier(client_out_ch, num_classes=10, pool_size=(2,2))).to(device) for _ in range(args.clients)]
        # load per-client states
        for i in range(len(ckpt['client_phis'])):
            if i < args.clients:
                client_phis[i].load_state_dict(ckpt['client_phis'][i])
                client_kappas[i].load_state_dict(ckpt['client_kappas'][i])

        # prepare datasets
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        rho_list = [float(x) for x in args.rho_list.split(',')]
        eth_list = [float(x) for x in args.eth_list.split(',')]

        # build results folder by model/dataset/method/rounds/lambda
        sweep_and_save_results(client_phis, client_kappas, global_theta, trainset, testset, args.clients, args.shards, args.shards_per_client, rho_list, eth_list, device,
                               model_name='vgg11', dataset_name='cifar10', method_name='splitgp', global_rounds=args.global_rounds, lam=args.lam,
                               out_base='results')

    # otherwise run federated training (this will also save a checkpoint)
    client_phis, client_kappas, global_theta = federated_train(
        vgg_factory_fn=make_vgg11_bn_for_cifar,
        num_clients=args.clients,
        num_shards=args.shards,
        shards_per_client=args.shards_per_client,
        global_rounds=args.global_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        device=device,
        save_path=args.save,
        rho=args.rho,
        eth=args.eth,
        num_workers=2,
        seed=0,
    )

    # Ensure we persist a final checkpoint (useful when auto-save is requested)
    try:
        if args.save:
            final_ckpt = {
                'global_theta': global_theta.state_dict(),
                'client_phis': [c.state_dict() for c in client_phis],
                'client_kappas': [k.state_dict() for k in client_kappas],
                'rounds': args.global_rounds,
                'avg_acc': None,
            }
            torch.save(final_ckpt, args.save)
            print(f"Saved final checkpoint to {args.save}")
    except Exception as e:
        print(f"Warning: failed to save final checkpoint to {getattr(args, 'save', None)}: {e}")

    # After training, run evaluation sweep using the trained models
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    rho_list = [float(x) for x in args.rho_list.split(',')]
    eth_list = [float(x) for x in args.eth_list.split(',')]

    # build results folder by model/dataset/method/rounds/lambda
    sweep_and_save_results(client_phis, client_kappas, global_theta, trainset, testset, args.clients, args.shards, args.shards_per_client, rho_list, eth_list, device,
                           model_name='vgg11', dataset_name='cifar10', method_name='splitgp', global_rounds=args.global_rounds, lam=args.lam,
                           out_base='results')

    print('Training and evaluation complete. Results saved as results.csv and results.png')


if __name__ == '__main__':
    main()

