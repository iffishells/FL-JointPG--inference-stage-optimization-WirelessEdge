# run_splitgp_vgg11.py
"""
Simplified SplitGP reproduction using VGG-11 backbone (CIFAR-10 by default).

- Implements:
  * Personalized local-only (each client trains full model locally)
  * FedAvg global (global full-model aggregated)
  * Split training (SplitGP family) with phi (client), kappa (client auxiliary), theta (server)
  * Multi-Exit baseline as SplitGP with lambda=0
- Produces CSV and plot of accuracy vs p (relative OOD fraction).
- Designed to be runnable on a single machine (GPU recommended).

Notes:
- This is simplified for clarity and speed. For a paper-scale run increase K, ROUNDS, and classifier sizes.
- The VGG classifier size is reduced by default to avoid OOM in small GPUs; you can toggle SMALL_CLASSIFIER=False to use original big layers.
"""
import os
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fontTools.misc.cython import returns
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse


# -------------------------
# Config (change as needed)
# -------------------------



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



def parse_args():
    parser = argparse.ArgumentParser(description="Run SplitGP/VGG11 experiments")

    # Experiment setup
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "FMNIST", "CIFAR10"],
                        help="Dataset to use")
    parser.add_argument("--clients", type=int, default=50, help="Number of clients (K)")
    parser.add_argument("--shards", type=int, default=100, help="Number of shards for non-IID split")
    parser.add_argument("--rounds", type=int, default=100, help="Number of global communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Epochs per client per round")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight between client-side and server-side loss")
    parser.add_argument("--lambda-splitgp", type=float, default=0.2, help="Personalization weight (lambda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Hardware
    parser.add_argument("--gpu", type=str, default="0", help="Which GPU id(s) to use (e.g. '0' or '0,1')")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoader")
    parser.add_argument("--eth", type=float, default=0.5, help="Entropy threshold (Eth) for selective offloading") # <--- ADD THIS LINE

    # Model toggles
    parser.add_argument("--small-classifier", action="store_true", help="Use smaller FC layers (avoid OOM)")
    parser.add_argument("--probe", action="store_true", help="Print probe feature shapes")

    # Method choice
    parser.add_argument("--method", type=str, default="splitgp",
                        choices=["splitgp", "multi-exit", "personalized", "fedavg", "all"],
                        help="Which training method to run")
    parser.add_argument("--split_index", type=int, default=11, help="SplitIndex") # <--- ADD THIS LINE



    return parser.parse_args()
# -------------------------
# Datasets helper
# -------------------------
def get_datasets(name="CIFAR10"):
    if name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        testset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
        in_channels, img_size = 1, 28
    elif name == "FMNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        testset  = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
        in_channels, img_size = 1, 28
    elif name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        testset  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        in_channels, img_size = 3, 32
    else:
        raise ValueError("Unknown dataset")
    return trainset, testset, in_channels, img_size

# -------------------------
# Non-IID shard partition
# -------------------------
def create_clients_shards(dataset, K=20, shards=100):
    n = len(dataset)
    idxs = np.arange(n)
    labels = np.array(dataset.targets)
    sorted_idx = idxs[np.argsort(labels)]
    shard_size = n // shards
    print(f"Shared size : {shard_size}")
    shard_idxs = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(shards)]
    print(f"number of shareds : {len(shard_idxs)}")
    random.shuffle(shard_idxs)
    clients = {}
    for i in range(K):
        clients[i] = np.concatenate(shard_idxs[2 * i:2 * i + 2])
    return clients

# -------------------------
# VGG-11 (small classifier option)
# -------------------------
class VGG11_small(nn.Module):
    """VGG-11 style network but with smaller FC layers (dev-friendly)."""
    def __init__(self, in_channels=3, num_classes=10, small_classifier=True):
        super().__init__()
        # features (same conv stack)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        if small_classifier:
            # smaller FCs: faster and less memory hungry
            self.classifier = nn.Sequential(
                nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(),
                nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(),
                nn.Linear(256, num_classes),
            )
        else:
            # original large classifier (paper-like) - may be heavy
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(),
                nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------
# Split modules: Phi, Theta, Kappa
# -------------------------
class Phi(nn.Module):
    """Client-side feature extractor: first N layers of vgg.features."""
    def __init__(self, vgg_model, split_index):
        super().__init__()
        self.net = nn.Sequential(*list(vgg_model.features.children())[:split_index])

    def forward(self, x):
        return self.net(x)

class Theta(nn.Module):
    """Server-side model: remaining feature layers + classifier (flatten between)."""
    def __init__(self, vgg_model, split_index):
        super().__init__()
        self.features_tail = nn.Sequential(*list(vgg_model.features.children())[split_index:])
        self.classifier = nn.Sequential(*list(vgg_model.classifier.children()))

    def forward(self, h):
        # h expected 4-D: (B,C,H,W)
        x = self.features_tail(h)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class Kappa(nn.Module):
    """Auxiliary classifier on top of phi features (flatten then linear)."""
    def __init__(self, feature_shape, num_classes=10):
        super().__init__()
        flat = int(np.prod(feature_shape))  # C*H*W
        self.fc = nn.Linear(flat, num_classes)

    def forward(self, h):
        return self.fc(torch.flatten(h, 1))

# -------------------------
# Utilities: data loaders, probe shapes, evaluation
# -------------------------
def probe_phi_feature_shape(phi_module, in_channels=3, img_size=32, device='cpu'):
    phi_module = phi_module.to(device)
    phi_module.eval()
    with torch.no_grad():
        sample = torch.randn(1, in_channels, img_size, img_size).to(device)
        feat = phi_module(sample)
        return tuple(feat.shape[1:])  # (C,H,W)

def get_client_loader(client_idx, batch=None):
    idxs = clients_indices[client_idx]
    subset = Subset(trainset, idxs)
    return DataLoader(subset, batch_size=batch, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

def client_test_set_for_p(client_idx, p):
    """
    Build local test set: all test samples from client's main classes + p * (#main) OOD samples.
    """
    train_idxs = clients_indices[client_idx]
    classes = np.unique(np.array(trainset.targets)[train_idxs])
    main_mask = np.isin(test_labels, classes)
    main_idxs = np.where(main_mask)[0]

    n_main = len(main_idxs)
    print(f"client {client_idx} has {n_main} main classes")

    n_ood = int(round(p * n_main))
    print(f"client {client_idx} will have {n_ood} OOD samples")

    non_main_idxs = np.where(~main_mask)[0]
    if n_ood > 0:
        replace = n_ood > len(non_main_idxs)
        ood_sample = np.random.choice(non_main_idxs, n_ood, replace=replace)
        final_idxs = np.concatenate([main_idxs, ood_sample])
    else:
        final_idxs = main_idxs
    return Subset(testset, final_idxs)

def evaluate_method_full_models(clients_state_dicts, per_client_testsets, in_channels, img_size, batch_size=None):
    """clients_state_dicts: list of CPU state_dicts for full models"""
    accs = []
    for k in range(K):
        state = clients_state_dicts[k]
        # rebuild full model
        vgg = VGG11_small(in_channels=in_channels, num_classes=10, small_classifier=SMALL_CLASSIFIER).to(DEVICE)
        vgg.load_state_dict(state)
        vgg.eval()
        correct = 0
        total = 0
        loader = DataLoader(per_client_testsets[k], batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                if not torch.is_tensor(yb):
                    yb = torch.tensor(yb, dtype=torch.long)
                yb = yb.to(DEVICE)
                out = vgg(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accs.append(correct / total if total > 0 else 0.0)
        del vgg
        torch.cuda.empty_cache()
    return float(np.mean(accs))

# def evaluate_method_split(clients_phi_states,
#                           clients_kappa_states,
#                           server_theta_state,
#                           per_client_testsets,
#                           in_channels,
#                           img_size,
#                           split_index,
#                           batch_size=None):
#     """
#     Evaluate split model:
#       - full model output computed by phi -> server_theta (server-side prediction)
#       - client-side prediction phi -> kappa
#     Return (avg_server_full_acc, avg_client_acc)
#     """
#     # rebuild server theta
#     base = VGG11_small(in_channels=in_channels, num_classes=10, small_classifier=SMALL_CLASSIFIER).to(DEVICE)
#     server_theta = Theta(base, split_index=split_index).to(DEVICE)
#     server_theta.load_state_dict({k: v.to(DEVICE) for k, v in server_theta_state.items()})
#     server_theta.eval()
#
#     full_accs = []
#     client_accs = []
#     for k in range(K):
#         # rebuild phi & kappa
#         base_local = VGG11_small(in_channels=in_channels, num_classes=10, small_classifier=SMALL_CLASSIFIER).to(DEVICE)
#         phi = Phi(base_local, split_index=split_index).to(DEVICE)
#         feat_shape = probe_phi_feature_shape(phi, in_channels=in_channels, img_size=img_size, device=DEVICE)
#         kappa = Kappa(feat_shape, num_classes=10).to(DEVICE)
#
#         phi.load_state_dict({kk: vv.to(DEVICE) for kk, vv in clients_phi_states[k].items()})
#         kappa.load_state_dict({kk: vv.to(DEVICE) for kk, vv in clients_kappa_states[k].items()})
#         phi.eval(); kappa.eval()
#
#         correct_full = 0
#         correct_client = 0
#         total = 0
#         threshold = 0.2
#         loader = DataLoader(per_client_testsets[k], batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
#         with torch.no_grad():
#             for xb, yb in loader:
#                 xb = xb.to(DEVICE)
#                 if not torch.is_tensor(yb):
#                     yb = torch.tensor(yb, dtype=torch.long)
#                 yb = yb.to(DEVICE)
#
#                 h = phi(xb)
#
#
#                 out_c = kappa(h).argmax(1)
#                 out_s = server_theta(h).argmax(1)
#                 correct_client += (out_c == yb).sum().item()
#                 correct_full += (out_s == yb).sum().item()
#                 total += yb.size(0)
#         client_accs.append(correct_client / total if total > 0 else 0.0)
#         full_accs.append(correct_full / total if total > 0 else 0.0)
#
#         del phi, kappa, base_local
#         torch.cuda.empty_cache()
#
#     del server_theta
#     torch.cuda.empty_cache()
#     return float(np.mean(full_accs)), float(np.mean(client_accs))




def evaluate_method_split(clients_phi_states,
                          clients_kappa_states,
                          server_theta_state,
                          per_client_testsets,
                          in_channels,
                          img_size,
                          split_index,
                          threshold,  # <--- NEW ARGUMENT (Eth)
                          batch_size=None):
    """
    Evaluate split model:
      - full model output computed by phi -> server_theta (server-side prediction)
      - client-side prediction phi -> kappa
    Return (avg_server_full_acc, avg_client_acc, avg_selective_acc)
    """
    # rebuild server theta (same as before)
    base = VGG11_small(in_channels=in_channels, num_classes=10, small_classifier=SMALL_CLASSIFIER).to(DEVICE)
    server_theta = Theta(base, split_index=split_index).to(DEVICE)
    server_theta.load_state_dict({k: v.to(DEVICE) for k, v in server_theta_state.items()})
    server_theta.eval()

    full_accs = []
    client_accs = []
    selective_accs = []  # <--- MODIFIED TO INCLUDE SELECTIVE

    for k in range(K):
        # ... (rebuild phi & kappa - same as before)
        base_local = VGG11_small(in_channels=in_channels, num_classes=10, small_classifier=SMALL_CLASSIFIER).to(DEVICE)
        phi = Phi(base_local, split_index=split_index).to(DEVICE)
        feat_shape = probe_phi_feature_shape(phi, in_channels=in_channels, img_size=img_size, device=DEVICE)
        kappa = Kappa(feat_shape, num_classes=10).to(DEVICE)

        phi.load_state_dict({kk: vv.to(DEVICE) for kk, vv in clients_phi_states[k].items()})
        kappa.load_state_dict({kk: vv.to(DEVICE) for kk, vv in clients_kappa_states[k].items()})
        phi.eval()
        kappa.eval()

        correct_full = 0
        correct_client = 0
        correct_selective = 0
        total = 0  # <--- ADD correct_selective
        loader = DataLoader(per_client_testsets[k], batch_size=batch_size, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                if not torch.is_tensor(yb):
                    yb = torch.tensor(yb, dtype=torch.long)
                yb = yb.to(DEVICE)

                h = phi(xb)

                # Get soft outputs from both paths
                out_c_soft = kappa(h)  # Client (kappa) soft output
                out_s_soft = server_theta(h)  # Server (theta) soft output

                # Calculate predictions
                pred_c = out_c_soft.argmax(dim=1)
                pred_s = out_s_soft.argmax(dim=1)

                # --- SplitGP Selective Offloading Logic ---
                # 1. Compute Shannon Entropy (E) for Client's prediction (Uncertainty)
                probs_c = F.softmax(out_c_soft, dim=1)
                # Shannon Entropy E = - sum(p * log2(p)). Add 1e-12 for numerical stability.
                entropy = -torch.sum(probs_c * torch.log2(probs_c + 1e-12), dim=1)
                # 2. Decision Mask: Offload if Uncertainty (E) >= Threshold (Eth)
                offload_mask = entropy >= threshold

                # 3. Final Prediction is selected based on the mask
                final_pred = torch.where(offload_mask, pred_s, pred_c)

                # Update Accuracy Counts
                correct_client += (pred_c == yb).sum().item()
                correct_full += (pred_s == yb).sum().item()
                correct_selective += (final_pred == yb).sum().item()  # <--- Selective Accuracy
                total += yb.size(0)

        client_accs.append(correct_client / total if total > 0 else 0.0)
        full_accs.append(correct_full / total if total > 0 else 0.0)
        selective_accs.append(correct_selective / total if total > 0 else 0.0)  # <--- Append Selective Acc

        del phi, kappa, base_local
        torch.cuda.empty_cache()

    del server_theta
    torch.cuda.empty_cache()
    # <--- RETURN SELECTIVE ACCURACY
    return float(np.mean(full_accs)), float(np.mean(client_accs)), float(np.mean(selective_accs))

def train_split_training(in_channels, img_size, split_index,
                         lambda_personalization=None,
                         gamma=None,
                         lr=None,
                         batch=None,
                         rounds=None,
                         client_loader=None):
    """
    SplitGP style training.
    Returns (clients_phi_states_cpu, clients_kappa_states_cpu, server_theta_state_cpu)
    """

    # sanity check
    if client_loader is None:
        raise ValueError("client_loaders must be provided (list or dict of DataLoaders indexed by client id)")

    # base model
    base = VGG11_small(in_channels=in_channels, num_classes=10, small_classifier=SMALL_CLASSIFIER).to(DEVICE)
    phi_template = Phi(base, split_index=split_index).to(DEVICE)
    theta_template = Theta(base, split_index=split_index).to(DEVICE)

    feat_shape = probe_phi_feature_shape(phi_template, in_channels=in_channels, img_size=img_size, device=DEVICE)
    # prepare client copies
    clients_phi = [deepcopy(phi_template).to(DEVICE) for _ in range(K)]
    clients_kappa = [Kappa(feat_shape, num_classes=10).to(DEVICE) for _ in range(K)]
    server_theta = deepcopy(theta_template).to(DEVICE)

    # initialize broadcast states (CPU)
    phi_state = {k: v.cpu().clone() for k, v in phi_template.state_dict().items()}
    kappa_state = {k: v.cpu().clone() for k, v in clients_kappa[0].state_dict().items()}
    theta_state = {k: v.cpu().clone() for k, v in theta_template.state_dict().items()}

    clients_phi_states_cpu = None
    clients_kappa_states_cpu = None

    for r in tqdm(range(rounds), desc="Global rounds (split)"):
        client_phi_states = []
        client_kappa_states = []
        client_theta_states = []

        for k in tqdm(range(K),desc=f" Clients training in rounds : {r}", leave=False):

            # load broadcast to client (move to device)
            clients_phi[k].load_state_dict({kk: vv.to(DEVICE) for kk, vv in phi_state.items()})
            clients_kappa[k].load_state_dict({kk: vv.to(DEVICE) for kk, vv in kappa_state.items()})
            server_theta.load_state_dict({kk: vv.to(DEVICE) for kk, vv in theta_state.items()})

            opt = optim.SGD(list(clients_phi[k].parameters()) +
                            list(clients_kappa[k].parameters()) +
                            list(server_theta.parameters()), lr=lr)

            # loader = get_client_loader(k, batch=batch)
            loader = client_loader[k]

            clients_phi[k].train()
            clients_kappa[k].train()
            server_theta.train()

            for _ in range(LOCAL_EPOCHS):
                for xb, yb in loader:
                    xb = xb.to(DEVICE)
                    if not torch.is_tensor(yb):
                        yb = torch.tensor(yb, dtype=torch.long)
                    yb = yb.to(DEVICE)
                    h = clients_phi[k](xb)                     # (B,C,H,W)
                    out_c = clients_kappa[k](h)
                    out_s = server_theta(h)
                    loss = gamma * F.cross_entropy(out_c, yb) + (1.0 - gamma) * F.cross_entropy(out_s, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            client_phi_states.append({kk: vv.cpu().clone() for kk, vv in clients_phi[k].state_dict().items()})
            client_kappa_states.append({kk: vv.cpu().clone() for kk, vv in clients_kappa[k].state_dict().items()})
            client_theta_states.append({kk: vv.cpu().clone() for kk, vv in server_theta.state_dict().items()})

            del opt
            torch.cuda.empty_cache()

        # aggregate server theta by FedAvg (CPU)
        new_theta = {}
        for key in client_theta_states[0].keys():
            s = sum(client_theta_states[i][key].to(torch.float32) for i in range(K))
            new_theta[key] = (s / K).clone()
        theta_state = new_theta

        # compute avg phi and avg kappa (CPU)
        avg_phi = {}
        for key in client_phi_states[0].keys():
            s = sum(client_phi_states[i][key].to(torch.float32) for i in range(K))
            avg_phi[key] = (s / K).clone()

        avg_kappa = {}
        for key in client_kappa_states[0].keys():
            s = sum(client_kappa_states[i][key].to(torch.float32) for i in range(K))
            avg_kappa[key] = (s / K).clone()

        # personalize: new client phi/kappa = lambda * local + (1-lambda) * avg
        new_client_phi_states = []
        new_client_kappa_states = []
        for i in range(K):
            local_phi = client_phi_states[i]
            local_kappa = client_kappa_states[i]
            new_phi = {}
            new_kappa = {}
            for key in local_phi.keys():
                new_phi[key] = (lambda_personalization * local_phi[key].to(torch.float32)
                                + (1 - lambda_personalization) * avg_phi[key]).clone()
            for key in local_kappa.keys():
                new_kappa[key] = (lambda_personalization * local_kappa[key].to(torch.float32)
                                  + (1 - lambda_personalization) * avg_kappa[key]).clone()
            new_client_phi_states.append(new_phi)
            new_client_kappa_states.append(new_kappa)

        # broadcast avg for next round initialization
        phi_state = {k: v.clone() for k, v in avg_phi.items()}
        kappa_state = {k: v.clone() for k, v in avg_kappa.items()}

        # store personalized states to return after training
        clients_phi_states_cpu = new_client_phi_states
        clients_kappa_states_cpu = new_client_kappa_states

    # final server theta CPU
    server_theta_state_cpu = {k: v.cpu().clone() for k, v in theta_state.items()}
    return clients_phi_states_cpu, clients_kappa_states_cpu, server_theta_state_cpu


def train_personalized_local_only(in_channels, img_size, lr, batch, rounds, client_loader):
    """
    Trains K separate full VGG-11 models locally for 'rounds' epochs (full personalization).
    No communication or aggregation takes place.
    Returns: list of K full model state dicts (CPU).
    """
    # Initialize K independent full VGG models
    clients_models = [
        VGG11_small(in_channels=in_channels, num_classes=10, small_classifier=SMALL_CLASSIFIER).to(DEVICE)
        for _ in range(K)
    ]

    # Initialize K independent optimizers
    clients_opts = [optim.SGD(model.parameters(), lr=lr) for model in clients_models]

    for r in tqdm(range(rounds), desc="Global rounds (Personalized Local)"):
        for k in tqdm(range(K), desc=" Clients Number ", leave=False):
            model = clients_models[k]
            opt = clients_opts[k]
            loader = client_loader[k]
            model.train()

            for _ in range(LOCAL_EPOCHS):
                for xb, yb in loader:
                    xb = xb.to(DEVICE)
                    if not torch.is_tensor(yb):
                        yb = torch.tensor(yb, dtype=torch.long)
                    yb = yb.to(DEVICE)

                    out = model(xb)
                    loss = F.cross_entropy(out, yb)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # Return final states of all K personalized models
    return [model.state_dict() for model in clients_models]


def train_fedavg(in_channels, img_size, lr, batch, rounds, client_loader):
    """
    Trains a single global VGG-11 model using Federated Averaging (FedAvg).
    Returns: a single global model state dict (CPU).
    """
    # 1. Initialize a single global model
    global_model = VGG11_small(in_channels=in_channels, num_classes=10, small_classifier=SMALL_CLASSIFIER).to(DEVICE)
    global_state = global_model.state_dict()

    for r in tqdm(range(rounds), desc="Global rounds (FedAvg)"):
        client_updates = []

        for k in tqdm(range(K), desc=" Clients Number ", leave=False):
            # 2. Local client training
            local_model = deepcopy(global_model).to(DEVICE)
            opt = optim.SGD(local_model.parameters(), lr=lr)
            loader = client_loader[k]
            local_model.train()

            for _ in range(LOCAL_EPOCHS):
                for xb, yb in loader:
                    xb = xb.to(DEVICE)
                    if not torch.is_tensor(yb):
                        yb = torch.tensor(yb, dtype=torch.long)
                    yb = yb.to(DEVICE)

                    out = local_model(xb)
                    loss = F.cross_entropy(out, yb)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            # 3. Collect update weights (delta_weights)
            # Since all data sizes are equal (2 shards per client), update aggregation is a simple average.
            client_updates.append(local_model.state_dict())

            del local_model, opt
            torch.cuda.empty_cache()

        # 4. Aggregation (FedAvg)
        new_global_state = {}
        for key in global_state.keys():
            # Average the weights across all clients
            s = sum(client_updates[i][key].to(torch.float32) for i in range(K))
            new_global_state[key] = (s / K).clone()

        global_model.load_state_dict(new_global_state)
        global_state = new_global_state  # Update state for next round

    # Return final global state
    return global_state
if __name__ == "__main__":

    args = parse_args()

    # Fix seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Setup GPU env
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)

    # Assign args to variables
    K = args.clients
    SHARDS = args.shards
    ROUNDS = args.rounds
    LOCAL_EPOCHS = args.local_epochs
    BATCH = args.batch_size
    LR = args.lr
    GAMMA = args.gamma
    LAMBDA_SPLITGP = args.lambda_splitgp
    SEED = args.seed
    NUM_WORKERS = args.num_workers
    PIN_MEMORY = not args.no_pin_memory
    SMALL_CLASSIFIER = args.small_classifier
    PROBE_PRINTS = args.probe
    DATASET = args.dataset
    ETH = args.eth # <--- GET ETH FROM ARGS
    split_index = args.split_index
    method = args.method
    results_folder_name = "results"
    os.makedirs(results_folder_name, exist_ok=True)

    p_values = [0.0 ,0.2,0.4,0.6,0.8,1.0]  # OOD fractions to evaluate
    OUT_DIR = f"{results_folder_name}/splitgp_vgg11_results_{DATASET}_method_{method}_rounds_{ROUNDS}_clients_{K}_model_split_index_{split_index}_gamma_{GAMMA}_lambda_split_{LAMBDA_SPLITGP}_ETH_{ETH}"
    print("Out dir",OUT_DIR)
    list_of_previous_experiments = os.listdir(OUT_DIR)

    if OUT_DIR in list_of_previous_experiments:
        print(f"Experiment with name {OUT_DIR} already exists. Please choose a different name.")
        exit(0)
    else:
        print(f"Experiment with name {OUT_DIR} will be created.")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Run methods ----
    methods_to_run = [args.method] if args.method != "all" else ["splitgp", "fedavg", "personalized", "multi-exit"]
    print("method to run: ", methods_to_run)
    trainset, testset, IN_CHANNELS, IMG_SIZE = get_datasets(DATASET)
    train_labels = np.array(trainset.targets)
    test_labels = np.array(testset.targets)

    clients_indices = create_clients_shards(trainset, K=K, shards=SHARDS)
    print(f"Created {K} clients; approx samples per client = {len(clients_indices[0])}")
    # Build client loaders once, reuse for all methods
    client_loaders = {i: get_client_loader(i, batch=BATCH) for i in range(K)}

    # Build per-client test sets for each p
    per_p_testsets = {}
    for p in p_values:
        per_p_testsets[p] = [client_test_set_for_p(k, p) for k in range(K)]

    # choose a split index for VGG features (must be consistent)
    # We pick split_index so phi contains first 6 conv/ReLU/pool blocks (tune if needed)
    # Using the features list length: VGG11_small.features children length -> we'll print probe
    tmp_base = VGG11_small(in_channels=IN_CHANNELS, num_classes=10, small_classifier=SMALL_CLASSIFIER)
    feat_children = list(tmp_base.features.children())
    print("Features children length:", len(feat_children))
    # choose split roughly so phi has 4 conv layers as in paper; select index by inspection
    # safe choice: split_index = 10 (used earlier) - this works in our construction

    # Optional: probe shapes
    phi_tmp = Phi(tmp_base, split_index=split_index)
    feat_shape = probe_phi_feature_shape(phi_tmp, in_channels=IN_CHANNELS, img_size=IMG_SIZE, device=DEVICE)
    if PROBE_PRINTS:
        print("Probe phi output feature shape (C,H,W):", feat_shape)

    results = defaultdict(list)
    all_sweep_results = []
    p_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lambda_values = [0.0, 0.2]
    gamma_values = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    eth_thresholds = [0.05, 0.1, 0.2, 0.8, 1.2, 1.6, 2.3, 2.5, 2.7, 2.8, 3.0]
    if "multi-exit" in methods_to_run:
        # 1) Multi-Exit baseline (lambda=0 -> clients fully replaced by avg)
        print("Training Multi-Exit (split, lambda=0) ...")
        clients_phi_me, clients_kappa_me, server_theta_me = train_split_training(
                                                                            in_channels=IN_CHANNELS,
                                                                            img_size=IMG_SIZE,
                                                                            split_index=split_index,
                                                                            lambda_personalization=0.0,
                                                                            gamma=GAMMA,
                                                                            lr=LR,
                                                                            batch=BATCH,
                                                                            rounds=ROUNDS,
                                                                            client_loader=client_loaders
                                                                            )
        for eth in eth_thresholds:
            multi_exit_current_result_set = {
                "p": [],
                "full_acc": [],
                "client_acc": [],
                "selective_acc": [],
                "gamma": [],
                "lambda": [],
                "eth": [],
            }
            for p in p_values:
                full_acc, client_acc, selective_acc = evaluate_method_split(clients_phi_me,
                                                             clients_kappa_me,
                                                             server_theta_me,
                                                             per_p_testsets[p],
                                                             in_channels=IN_CHANNELS,
                                                             img_size=IMG_SIZE,
                                                             split_index=split_index,
                                                             threshold=eth,
                                                             batch_size=BATCH)
                multi_exit_current_result_set["p"].append(p)
                multi_exit_current_result_set['full_acc'].append(full_acc)
                multi_exit_current_result_set["client_acc"].append(client_acc)
                multi_exit_current_result_set["selective_acc"].append(selective_acc)
                multi_exit_current_result_set["gamma"].append(GAMMA)
                multi_exit_current_result_set["lambda"].append(LAMBDA_SPLITGP)
                multi_exit_current_result_set["eth"].append(eth)

                print(f"p={p:.2f} Multi-Exit full acc: {full_acc:.4f}  client acc: {client_acc:.4f}, selective_acc : {selective_acc:.4f} ")

            df = pd.DataFrame(multi_exit_current_result_set, index=p_values)
            df.index.name = 'p'
            csv_name = f"{method}_combined_results_eth_{eth:.2f}_gamma_{GAMMA}_lambda_split_{LAMBDA_SPLITGP}.csv"  # 👈 separate CSV per Eth
            csv_path = os.path.join(OUT_DIR, csv_name)
            df.to_csv(csv_path)
            print("Saved CSV ->", csv_path)



    # sudo testing
    # lambda_values = [ 0.5]
    # gamma_values = [0.5]
    # eth_thresholds = [2.5]
    if "splitgp" in methods_to_run:

        # 2) SplitGP (lambda = LAMBDA_SPLITGP)
        print("Training SplitGP (lambda=%.2f) ..." % (LAMBDA_SPLITGP))
        for LAMBDA_SPLITGP in lambda_values:

            # Outer loop: Sweep Gamma (Training Hyperparameter)
            for GAMMA in gamma_values :
                print(f"\n--- Training with GAMMA={GAMMA:.2f} and lambda : {LAMBDA_SPLITGP:.2f} ---")

                # 1. Train the model for this specific GAMMA value
                clients_phi_sgp, clients_kappa_sgp, server_theta_sgp = train_split_training(
                    in_channels=IN_CHANNELS,
                    img_size=IMG_SIZE,
                    split_index=split_index,
                    lambda_personalization=LAMBDA_SPLITGP,
                    gamma=GAMMA,  # Use current GAMMA
                    lr=LR,
                    batch=BATCH,
                    rounds=ROUNDS,
                    client_loader=client_loaders)

                # Inner loop: Sweep Eth (Inference Hyperparameter)

                for eth in eth_thresholds:
                    print(f"\n>>> Evaluating SplitGP (G={GAMMA:.2f}) with Eth={eth:.2f}")

                    # Temporary dict to hold results for this single (GAMMA, Eth) combination across all p_values
                    current_result_set = {
                        "p": [],
                        "full_acc": [],
                        "client_acc": [],
                        "selective_acc": [],
                        "gamma": [],
                        "lambda": [],
                        "eth": [],
                    }

                    for p in p_values:
                        # 2. Evaluate the trained model for the current Eth and p_value
                        print(f"Evaluating p : {p} and ETH : {eth} and gamma : {GAMMA} and lambda : {LAMBDA_SPLITGP}")
                        full_acc, client_acc, selective_acc = evaluate_method_split(
                            clients_phi_sgp,
                            clients_kappa_sgp,
                            server_theta_sgp,
                            per_p_testsets[p],
                            in_channels=IN_CHANNELS,
                            img_size=IMG_SIZE,
                            split_index=split_index,
                            threshold=eth,  # Use current Eth
                            batch_size=BATCH)

                        # Record results into the temporary dictionary
                        current_result_set["p"].append(p)
                        current_result_set['full_acc'].append(full_acc)
                        current_result_set["client_acc"].append(client_acc)
                        current_result_set["selective_acc"].append(selective_acc)
                        current_result_set["gamma"].append(GAMMA)
                        current_result_set["lambda"].append(LAMBDA_SPLITGP)
                        current_result_set["eth"].append(eth)

                        print(
                            f"G={GAMMA:.2f} |Lambda : {LAMBDA_SPLITGP} | Eth={eth:.2f} | p={p:.2f} | Full Acc: {full_acc:.4f} | Client Acc: {client_acc:.4f} | Selective Acc: {selective_acc:.4f}"
                        )

                    df_temp = pd.DataFrame(current_result_set)
                    df = pd.DataFrame(current_result_set, index=p_values)
                    df.index.name = 'p'
                    csv_name = f"{method}_combined_results_eth_{eth:.2f}_gamma_{GAMMA}_lambda_split_{LAMBDA_SPLITGP}.csv"  # 👈 separate CSV per Eth
                    csv_path = os.path.join(OUT_DIR, csv_name)
                    df.to_csv(csv_path)
                    print("Saved CSV ->", csv_path)

                    # Append this DataFrame's data to the overall list
                    all_sweep_results.append(df_temp)

        # 3. Combine all results and save a single, comprehensive CSV
        if all_sweep_results:
            # Concatenate all DataFrames into one large table
            df_final = pd.concat(all_sweep_results, ignore_index=True)
            csv_name = "splitgp_gamma_eth_sweep_combined.csv"
            csv_path = os.path.join(OUT_DIR, csv_name)
            df_final.to_csv(csv_path, index=False)
            print("\n========================================================")
            print(f"✅ SUCCESSFULLY SAVED FINAL SWEEP RESULTS to -> {csv_path}")
            print("========================================================")

    if "personalized" in methods_to_run:
        print("Personalized (local-only) training ...")
        # 3) Personalized local-only baseline (full models per client)
        print("Training Personalized (local-only) models ...")
        personal_states = train_personalized_local_only(in_channels=IN_CHANNELS,
                                                       img_size=IMG_SIZE,
                                                       batch=BATCH,
                                                       rounds=ROUNDS,
                                                       lr=LR,
                                                       client_loader=client_loaders)
        # Evaluate each client's unique full model
        for p in p_values:
            # Note: This uses the existing evaluation helper for K full models.
            avg_acc = evaluate_method_full_models(personal_states, per_p_testsets[p], in_channels=IN_CHANNELS,
                                                  img_size=IMG_SIZE, batch_size=BATCH)
            results['Personalized'].append(avg_acc)
            print(f"p={p:.2f} Personalized avg acc: {avg_acc:.4f}")
        pd.DataFrame({'Personalized': results['Personalized']}, index=p_values).to_csv(
            os.path.join(OUT_DIR, 'personalized.csv'))

    if "fedavg" in methods_to_run:
        print("Training FedAvg ...")
        # 4) FedAvg global baseline (Generalized Global Model via FL)
        print("Training FedAvg (Generalized Global) model ...")
        global_state = train_fedavg(in_channels=IN_CHANNELS,
                                    img_size=IMG_SIZE,
                                    batch=BATCH,
                                    rounds=ROUNDS,
                                    lr=LR,
                                    client_loader=client_loaders)

        # Prepare a list of K identical states for the evaluation helper
        global_states_list = [global_state] * K

        # Evaluate the single global model on all K test sets
        for p in p_values:
            # Note: This uses the evaluation helper for K full models, but they are all identical.
            avg_acc = evaluate_method_full_models(global_states_list, per_p_testsets[p], in_channels=IN_CHANNELS,
                                                  img_size=IMG_SIZE, batch_size=BATCH)
            results['FedAvg Global'].append(avg_acc)
            print(f"p={p:.2f} FedAvg Global avg acc: {avg_acc:.4f}")

        pd.DataFrame({'FedAvg Global': results['FedAvg Global']}, index=p_values).to_csv(
            os.path.join(OUT_DIR, 'fedavg_global.csv'))


    # Plot
    print("Plotting results ...")
    plt.figure(figsize=(8,6))
    for method, accs in results.items():
        plt.plot(p_values, accs, marker='o', label=method)
    plt.xlabel("p (relative portion of OOD samples)")
    plt.ylabel("Test accuracy")
    plt.title(f"{method}: Accuracy vs p")
    plt.xticks(p_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "accuracy_vs_p.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, "accuracy_vs_p.pdf"))
    plt.show()
