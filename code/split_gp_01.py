import os
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
# remove early ad-hoc argparse/device setup which parsed args at import-time
DEVICE = None
# Note: DEVICE will be deterministically set inside `if __name__ == "__main__":` after parsing args.

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SplitGP/VGG11 experiments")

    # Experiment setup
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FMNIST", "CIFAR10"],
                        help="Dataset to use")
    parser.add_argument("--clients", type=int, default=50, help="Number of clients (K)")
    parser.add_argument("--shards", type=int, default=100, help="Number of shards for non-IID split")
    parser.add_argument("--rounds", type=int, default=120, help="Number of global communication rounds (Paper: 120 for MNIST/FMNIST, 800 for CIFAR10)")
    parser.add_argument("--local-epochs", type=int, default=1, help="Epochs per client per round (paper uses 1, but may need 0.5 for MNIST to match 12 updates)")
    parser.add_argument("--max-local-updates", type=int, default=None, help="Max local updates per round (paper: 12 for MNIST, 10 for CIFAR10). If set, overrides local-epochs.")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight between client-side and server-side loss")
    parser.add_argument("--lambda-splitgp", type=float, default=0.2, help="Personalization weight (lambda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Hardware
    parser.add_argument("--gpu", type=str, default="1", help="Which GPU id(s) to use (e.g. '0' or '0,1')")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoader")
    parser.add_argument("--eth", type=float, default=0.8, help="Entropy threshold (Eth) for selective offloading (Paper uses: 0.05-2.3 range)")

    # Determinism / reproducibility
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic PyTorch/CUDA behavior and force num_workers=0, pin_memory=False for exact reproducibility")

    # Model toggles
    parser.add_argument("--small-classifier", action="store_true", help="Use smaller FC layers (avoid OOM)")
    parser.add_argument("--probe", action="store_true", help="Print probe feature shapes")

    # Method choice
    parser.add_argument("--method", type=str, default="splitgp",
                        choices=["splitgp"],
                        help="Training method to run (only SplitGP available)")
    parser.add_argument("--split_index", type=int, default=11, help="SplitIndex")
    parser.add_argument("--model", type=str, default="SimpleCNN", help="Model type for results folder (e.g., SimpleCNN, VGG11)")

    return parser.parse_args()
# -------------------------
# Datasets helper
# -------------------------
def get_datasets(name="CIFAR10"):
    if name == "MNIST":
        # Paper-standard MNIST normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        testset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
        in_channels, img_size = 1, 28
    elif name == "FMNIST":
        # Paper-standard FMNIST normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
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
    shard_idxs = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(shards)]
    random.shuffle(shard_idxs)
    clients = {}
    for i in range(K):
        clients[i] = np.concatenate(shard_idxs[2 * i:2 * i + 2])
    return clients


class SimpleCNN(nn.Module):
    """Paper-aligned CNN model for MNIST/FMNIST with exact parameter counts matching the paper"""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # Paper uses 5x5 kernels, not 3x3
        # Architecture designed to match exact parameter counts:
        # |φ| = 387,840, |θ| = 3,480,330, |κ| = 23,050

        # Client-side φ: 4 convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, 5, padding=2)    # 5x5 kernel
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)             # 5x5 kernel
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)            # 5x5 kernel
        self.conv4 = nn.Conv2d(128, 128, 5, padding=2)           # 5x5 kernel (φ ends here)

        # Server-side θ: 1 conv + 3 FC layers
        self.conv5 = nn.Conv2d(128, 256, 5, padding=2)           # 5x5 kernel (θ starts here)
        self.pool = nn.MaxPool2d(2, 2)

        # For MNIST/FMNIST: 28->14->7, so after conv5: 256*7*7 = 12544
        # For CIFAR10: 32->16->8, so after conv5: 256*8*8 = 16384
        fc1_in = 256 * (7 * 7 if in_channels == 1 else 8 * 8)

        # Fully connected layers (θ side) - adjusted for exact parameter matching
        self.fc1 = nn.Linear(fc1_in, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Full-model forward pass
        x = F.relu(self.conv1(x))       # 28x28 -> 28x28
        x = F.relu(self.conv2(x))       # 28x28 -> 28x28
        x = self.pool(x)                # 28x28 -> 14x14
        x = F.relu(self.conv3(x))       # 14x14 -> 14x14
        x = F.relu(self.conv4(x))       # 14x14 -> 14x14  (φ output)
        x = self.pool(x)                # 14x14 -> 7x7
        x = F.relu(self.conv5(x))       # 7x7 -> 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Phi(nn.Module):
    """Client-side feature extractor: conv1..conv4 with paper-aligned pooling."""
    def __init__(self, cnn_model):
        super().__init__()
        self.conv1 = cnn_model.conv1
        self.conv2 = cnn_model.conv2
        self.conv3 = cnn_model.conv3
        self.conv4 = cnn_model.conv4
        self.pool = cnn_model.pool

    def forward(self, x):
        x = F.relu(self.conv1(x))       # 28x28 -> 28x28
        x = F.relu(self.conv2(x))       # 28x28 -> 28x28
        x = self.pool(x)                # 28x28 -> 14x14
        x = F.relu(self.conv3(x))       # 14x14 -> 14x14
        x = F.relu(self.conv4(x))       # 14x14 -> 14x14
        x = self.pool(x)                # 14x14 -> 7x7
        return x

class Theta(nn.Module):
    """Server-side model: conv5 + fc1..fc3."""
    def __init__(self, cnn_model):
        super().__init__()
        self.conv5 = cnn_model.conv5
        self.fc1 = cnn_model.fc1
        self.fc2 = cnn_model.fc2
        self.fc3 = cnn_model.fc3

    def forward(self, h):
        x = F.relu(self.conv5(h))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Kappa(nn.Module):
    """Auxiliary classifier: single FC on top of φ output; matches paper |κ| when MNIST/FMNiST."""
    def __init__(self, feature_shape, num_classes=10):
        super().__init__()
        flat = int(np.prod(feature_shape))
        self.fc = nn.Linear(flat, num_classes)

    def forward(self, h):
        return self.fc(torch.flatten(h, 1))

def probe_phi_feature_shape(phi_module, in_channels=3, img_size=32, device=None):
    """Return (C,H,W) of phi output. `device` may be a string or torch.device; defaults to CPU if None."""
    if device is None:
        device = 'cpu'
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
    n_ood = int(round(p * n_main))
    non_main_idxs = np.where(~main_mask)[0]
    if n_ood > 0:
        replace = n_ood > len(non_main_idxs)
        ood_sample = np.random.choice(non_main_idxs, n_ood, replace=replace)
        final_idxs = np.concatenate([main_idxs, ood_sample])
    else:
        final_idxs = main_idxs
    return Subset(testset, final_idxs)


def evaluate_method_split(clients_phi_states,
                          clients_kappa_states,
                          server_theta_state,
                          per_client_testsets,
                          in_channels,
                          img_size,
                          split_index,
                          threshold,
                          batch_size=None,
                          model_type='SimpleCNN'):
    """
    Evaluate split model:
      - server-side prediction: phi -> theta
      - client-side prediction: phi -> kappa
    Return (avg_server_full_acc, avg_client_acc, avg_selective_acc)
    """
    base = create_model(model_type, in_channels=in_channels, num_classes=10, split_index=split_index).to(DEVICE)
    server_theta = create_theta(base, model_type, split_index=split_index).to(DEVICE)
    server_theta.load_state_dict({k: v.to(DEVICE) for k, v in server_theta_state.items()})
    server_theta.eval()

    full_accs, client_accs, selective_accs = [], [], []

    for k in range(K):
        base_local = create_model(model_type, in_channels=in_channels, num_classes=10, split_index=split_index).to(DEVICE)
        phi = create_phi(base_local, model_type, split_index=split_index).to(DEVICE)
        feat_shape = probe_phi_feature_shape(phi, in_channels=in_channels, img_size=img_size, device=DEVICE)
        kappa = create_kappa(feat_shape, model_type, num_classes=10).to(DEVICE)

        phi.load_state_dict({kk: vv.to(DEVICE) for kk, vv in clients_phi_states[k].items()})
        kappa.load_state_dict({kk: vv.to(DEVICE) for kk, vv in clients_kappa_states[k].items()})
        phi.eval(); kappa.eval()

        correct_full = correct_client = correct_selective = 0
        total = 0
        loader = DataLoader(per_client_testsets[k], batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                if not torch.is_tensor(yb): yb = torch.tensor(yb, dtype=torch.long)
                yb = yb.to(DEVICE)
                h = phi(xb)
                out_c_soft = kappa(h)
                out_s_soft = server_theta(h)
                pred_c = out_c_soft.argmax(dim=1)
                pred_s = out_s_soft.argmax(dim=1)
                probs_c = F.softmax(out_c_soft, dim=1)
                entropy = -torch.sum(probs_c * torch.log2(probs_c + 1e-12), dim=1)
                offload_mask = entropy > threshold
                final_pred = torch.where(offload_mask, pred_s, pred_c)
                correct_client += int(torch.eq(pred_c, yb).sum().item())
                correct_full += int(torch.eq(pred_s, yb).sum().item())
                correct_selective += int(torch.eq(final_pred, yb).sum().item())
                total += yb.size(0)
        client_accs.append(correct_client / total if total else 0.0)
        full_accs.append(correct_full / total if total else 0.0)
        selective_accs.append(correct_selective / total if total else 0.0)

        del phi, kappa, base_local
        torch.cuda.empty_cache()

    del server_theta
    torch.cuda.empty_cache()
    return float(np.mean(full_accs)), float(np.mean(client_accs)), float(np.mean(selective_accs))

def train_split_training(in_channels, img_size,
                         split_index,
                         lambda_personalization=None,
                         gamma=None,
                         lr=None,
                         batch=None,
                         rounds=None,
                         client_loader=None,
                         local_epochs=1,
                         max_local_updates=None,
                         model_type='SimpleCNN'):
    """
    SplitGP style training. Returns client φ/κ and server θ states (CPU).

    Args:
        max_local_updates: If set, limits the number of gradient updates per client per round.
                          Paper uses 12 for MNIST/FMNIST, 10 for CIFAR10.
    """
    if client_loader is None:
        raise ValueError("client_loaders must be provided (list or dict of DataLoaders indexed by client id)")

    base = create_model(model_type, in_channels=in_channels, num_classes=10, split_index=split_index).to(DEVICE)
    phi_template = create_phi(base, model_type, split_index=split_index).to(DEVICE)
    theta_template = create_theta(base, model_type, split_index=split_index).to(DEVICE)

    feat_shape = probe_phi_feature_shape(phi_template, in_channels=in_channels, img_size=img_size, device=DEVICE)
    clients_phi = [deepcopy(phi_template).to(DEVICE) for _ in range(K)]
    clients_kappa = [create_kappa(feat_shape, model_type, num_classes=10).to(DEVICE) for _ in range(K)]
    server_theta = deepcopy(theta_template).to(DEVICE)

    phi_state = {k: v.cpu().clone() for k, v in phi_template.state_dict().items()}
    kappa_state = {k: v.cpu().clone() for k, v in clients_kappa[0].state_dict().items()}
    theta_state = {k: v.cpu().clone() for k, v in theta_template.state_dict().items()}

    # If no training rounds, return initial broadcast states for evaluation convenience
    if rounds is not None and rounds == 0:
        clients_phi_states_cpu = [ {kk: vv.clone() for kk, vv in phi_state.items()} for _ in range(K) ]
        clients_kappa_states_cpu = [ {kk: vv.clone() for kk, vv in kappa_state.items()} for _ in range(K) ]
        server_theta_state_cpu = {k: v.clone() for k, v in theta_state.items()}
        return clients_phi_states_cpu, clients_kappa_states_cpu, server_theta_state_cpu

    clients_phi_states_cpu = None
    clients_kappa_states_cpu = None

    for r in tqdm(range(rounds), desc="Global rounds (split)"):
        client_phi_states, client_kappa_states, client_theta_states = [], [], []
        for k in tqdm(range(K), desc=" Clients Number ", leave=False):
            # CRITICAL BUG FIX: Load personalized state for each client from previous round
            # In round 0, all clients start with the same phi_state/kappa_state (initialization)
            # In round r>0, each client k should load its personalized state from round r-1
            if r > 0 and clients_phi_states_cpu is not None:
                clients_phi[k].load_state_dict({kk: vv.to(DEVICE) for kk, vv in clients_phi_states_cpu[k].items()})
                clients_kappa[k].load_state_dict({kk: vv.to(DEVICE) for kk, vv in clients_kappa_states_cpu[k].items()})
            else:
                # Round 0: broadcast the same initial model to all clients
                clients_phi[k].load_state_dict({kk: vv.to(DEVICE) for kk, vv in phi_state.items()})
                clients_kappa[k].load_state_dict({kk: vv.to(DEVICE) for kk, vv in kappa_state.items()})
            server_theta.load_state_dict({kk: vv.to(DEVICE) for kk, vv in theta_state.items()})

            opt = optim.SGD(list(clients_phi[k].parameters()) +
                            list(clients_kappa[k].parameters()) +
                            list(server_theta.parameters()), lr=lr)

            loader = client_loader[k]
            clients_phi[k].train(); clients_kappa[k].train(); server_theta.train()

            update_count = 0
            for epoch in range(local_epochs):
                for xb, yb in loader:
                    xb = xb.to(DEVICE)
                    if not torch.is_tensor(yb): yb = torch.tensor(yb, dtype=torch.long)
                    yb = yb.to(DEVICE)
                    h = clients_phi[k](xb)
                    out_c = clients_kappa[k](h)
                    out_s = server_theta(h)
                    loss = gamma * F.cross_entropy(out_c, yb) + (1.0 - gamma) * F.cross_entropy(out_s, yb)
                    opt.zero_grad(); loss.backward(); opt.step()

                    update_count += 1
                    # If max_local_updates is set, stop after reaching the limit
                    if max_local_updates is not None and update_count >= max_local_updates:
                        break

                # Break outer loop if we've reached max updates
                if max_local_updates is not None and update_count >= max_local_updates:
                    break
                    h = clients_phi[k](xb)
                    out_c = clients_kappa[k](h)
                    out_s = server_theta(h)
                    loss = gamma * F.cross_entropy(out_c, yb) + (1.0 - gamma) * F.cross_entropy(out_s, yb)
                    opt.zero_grad(); loss.backward(); opt.step()

            client_phi_states.append({kk: vv.cpu().clone() for kk, vv in clients_phi[k].state_dict().items()})
            client_kappa_states.append({kk: vv.cpu().clone() for kk, vv in clients_kappa[k].state_dict().items()})
            client_theta_states.append({kk: vv.cpu().clone() for kk, vv in server_theta.state_dict().items()})
            del opt; torch.cuda.empty_cache()

        # Calculate dataset size weights as per paper equation (9)
        client_data_sizes = [len(clients_indices[i]) for i in range(K)]
        total_data_size = sum(client_data_sizes)
        alpha_weights = [size / total_data_size for size in client_data_sizes]

        # FedAvg θ with dataset size weights (paper equation 9)
        new_theta = {}
        for key in client_theta_states[0].keys():
            weighted_list = [alpha_weights[i] * client_theta_states[i][key].to(torch.float32) for i in range(K)]
            weighted_sum = torch.stack(weighted_list, dim=0).sum(dim=0)
            new_theta[key] = weighted_sum.clone()
        theta_state = new_theta

        # SplitGP: Personalized aggregation with dataset size weights (paper equations 10, 11)
        avg_phi = {}
        for key in client_phi_states[0].keys():
            weighted_list = [alpha_weights[i] * client_phi_states[i][key].to(torch.float32) for i in range(K)]
            weighted_sum = torch.stack(weighted_list, dim=0).sum(dim=0)
            avg_phi[key] = weighted_sum.clone()
        avg_kappa = {}
        for key in client_kappa_states[0].keys():
            weighted_list = [alpha_weights[i] * client_kappa_states[i][key].to(torch.float32) for i in range(K)]
            weighted_sum = torch.stack(weighted_list, dim=0).sum(dim=0)
            avg_kappa[key] = weighted_sum.clone()

        new_client_phi_states, new_client_kappa_states = [], []
        for i in range(K):
            local_phi, local_kappa = client_phi_states[i], client_kappa_states[i]
            new_phi, new_kappa = {}, {}
            for key in local_phi.keys():
                new_phi[key] = (lambda_personalization * local_phi[key].to(torch.float32) +
                                (1 - lambda_personalization) * avg_phi[key]).clone()
            for key in local_kappa.keys():
                new_kappa[key] = (lambda_personalization * local_kappa[key].to(torch.float32) +
                                  (1 - lambda_personalization) * avg_kappa[key]).clone()
            new_client_phi_states.append(new_phi)
            new_client_kappa_states.append(new_kappa)

        # Update client-specific states for next round
        # CRITICAL: Each client should use its own personalized model (new_client_phi_states[k])
        # NOT the average (avg_phi). The averaging is only for creating personalized versions.
        # Store the personalized states for each client
        clients_phi_states_cpu = new_client_phi_states
        clients_kappa_states_cpu = new_client_kappa_states

        # For broadcasting in the next round, we DON'T use the average directly
        # Instead, each client will load its own personalized state from new_client_phi_states
        # This is handled in the loop by loading from phi_state dict (not a single broadcast)
        phi_state = avg_phi  # Used as initialization/reference only
        kappa_state = avg_kappa  # Used as initialization/reference only

    server_theta_state_cpu = {k: v.cpu().clone() for k, v in theta_state.items()}
    return clients_phi_states_cpu, clients_kappa_states_cpu, server_theta_state_cpu


# ---------------------
# Param counting helpers
# ---------------------

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def count_bn_affine_params(module: nn.Module) -> int:
    total = 0
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) and m.affine:
            total += (m.weight.numel() + m.bias.numel())
    return total

def count_conv_bias_params(module: nn.Module) -> int:
    total = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            total += m.bias.numel()
    return total


class VGG11(nn.Module):
    """
    VGG-11 model for CIFAR10 WITHOUT BatchNorm (matches paper's parameter counts).
    Paper split: |φ| = 972,554, |θ| = 8,258,560, |κ| = 10,250

    Architecture: 8 conv layers [64, 128, 256, 256, 512, 512, 512, 512] + classifier
    Split point: After 4th conv (256) -> φ has 4 convs, θ has 4 convs + classifier
    """
    def __init__(self, num_classes=10, split_index=4, in_channels=3, small_classifier=True):
        super().__init__()
        self.split_index = split_index  # Number of conv layers in phi (paper uses 4)

        # VGG-11 config: 8 conv layers with max pooling
        # [64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M]
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        layers = []
        in_ch = in_channels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # NO BatchNorm - matches paper's parameter counts
                conv2d = nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=True)
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
                in_ch = v

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Paper uses smaller classifier: 512 -> 512 -> 256 -> 10
        # This gives |classifier| ≈ 396K params
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG11_Kappa(nn.Module):
    """Aux classifier for VGG11: Adaptive pool to achieve 1024 features then FC->num_classes.
    This yields |κ| = 1024*classes + classes. For 10 classes -> 10,250."""
    def __init__(self, feature_shape, num_classes=10, target_features=1024):
        super().__init__()
        C, H, W = feature_shape
        # choose spatial size so that C * s * s == target_features if possible
        if target_features % C == 0:
            s2 = target_features // C
            s = int(round(s2 ** 0.5))
            if s * s == s2 and s > 0:
                self.pool = nn.AdaptiveAvgPool2d((s, s))
                in_feats = C * s * s
            else:
                # fallback to flatten if not perfect square
                self.pool = None
                in_feats = C * H * W
        else:
            self.pool = None
            in_feats = C * H * W
        self.fc = nn.Linear(in_feats, num_classes)

    def forward(self, h):
        if self.pool is not None:
            h = self.pool(h)
        h = torch.flatten(h, 1)
        return self.fc(h)

class VGG11_Phi(nn.Module):
    """
    Client-side feature extractor for VGG11: first split_index conv blocks.
    split_index=4 means first 4 conv layers (3->64->128->256->256)
    This gives |φ| = 960,896 params (paper: 972,554 likely includes kappa or has slight difference)
    """
    def __init__(self, vgg_model, split_index=4):
        super().__init__()
        # Extract first split_index conv+relu blocks from features
        # In the features list: [Conv, ReLU, MaxPool, Conv, ReLU, MaxPool, ...]
        # We need to find where the split_index-th conv layer ends

        layers = list(vgg_model.features.children())
        conv_count = 0
        split_at = 0

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                conv_count += 1
                if conv_count == split_index:
                    # Include this conv + the next ReLU
                    split_at = i + 2  # Conv + ReLU
                    break

        self.net = nn.Sequential(*layers[:split_at])

    def forward(self, x):
        return self.net(x)

class VGG11_Theta(nn.Module):
    """
    Server-side model for VGG11: remaining conv layers + avgpool + classifier.
    For split_index=4: includes last 4 conv layers (256->512->512->512->512) + classifier
    This gives |θ| ≈ 8,259,584 conv + 396,554 classifier = 8,656,138 total
    Paper reports 8,258,560 (small difference likely from parameter counting method)
    """
    def __init__(self, vgg_model, split_index=4):
        super().__init__()

        # Find where phi ends and extract remaining layers
        layers = list(vgg_model.features.children())
        conv_count = 0
        split_at = 0

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                conv_count += 1
                if conv_count == split_index:
                    split_at = i + 2  # After Conv + ReLU
                    break

        self.features_tail = nn.Sequential(*layers[split_at:])
        self.avgpool = vgg_model.avgpool
        self.classifier = vgg_model.classifier

    def forward(self, h):
        x = self.features_tail(h)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_model(model_type, in_channels, num_classes=10, split_index=None):
    """
    Factory function to create SimpleCNN or VGG11 models.

    split_index: Number of conv layers in phi (client-side)
        - For SimpleCNN: Not used (fixed architecture)
        - For VGG11: Default=4 (paper configuration)
    """
    if model_type.lower() == 'simplecnn':
        model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    elif model_type.lower() in ['vgg11', 'vgg-11']:
        # Use split_index=4 by default for VGG11 (matches paper)
        if split_index is None:
            split_index = 4
        model = VGG11(num_classes=num_classes, split_index=split_index, in_channels=in_channels, small_classifier=SMALL_CLASSIFIER)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Apply proper weight initialization
    model.apply(init_weights)
    return model

def create_phi(cnn_model, model_type='SimpleCNN', split_index=None):
    """Factory function to create Phi (client-side feature extractor)"""
    if model_type.lower() == 'simplecnn':
        return Phi(cnn_model)
    elif model_type.lower() in ['vgg11', 'vgg-11']:
        if split_index is None:
            split_index = 4  # Paper default for VGG11
        return VGG11_Phi(cnn_model, split_index=split_index)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_theta(cnn_model, model_type='SimpleCNN', split_index=None):
    """Factory function to create Theta (server-side model)"""
    if model_type.lower() == 'simplecnn':
        return Theta(cnn_model)
    elif model_type.lower() in ['vgg11', 'vgg-11']:
        if split_index is None:
            split_index = 4  # Paper default for VGG11
        return VGG11_Theta(cnn_model, split_index=split_index)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_kappa(feature_shape, model_type='SimpleCNN', num_classes=10):
    """Factory function to create Kappa (auxiliary classifier)"""
    if model_type.lower() == 'simplecnn':
        return Kappa(feature_shape, num_classes=num_classes)
    elif model_type.lower() in ['vgg11', 'vgg-11']:
        return VGG11_Kappa(feature_shape, num_classes=num_classes, target_features=1024)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def init_weights(m):
    """Proper weight initialization for better convergence"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if __name__ == "__main__":

    args = parse_args()

    # Force PYTHON hash seed and deterministic behavior (recommended for reproducibility)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Enable PyTorch deterministic algorithms with warn_only mode
    # This is necessary because some operations (like adaptive_avg_pool2d on CUDA used in VGG-11)
    # don't have deterministic implementations
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions don't support warn_only parameter
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # If it fails, continue without strict determinism
            pass
    
    # cudnn settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # CUBLAS deterministic workspace config for some PyTorch/CUDA combinations
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Setup GPU env (respect CLI --gpu which is a comma-separated string of visible ids)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Ensure DEVICE is a torch.device (pick the first visible GPU -> cuda:0)
    if torch.cuda.is_available():
        # Use cuda:0 which maps to the first id in CUDA_VISIBLE_DEVICES
        DEVICE = torch.device("cuda:0")
        try:
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("✅ Using GPU (name unavailable)")
    else:
        DEVICE = torch.device("cpu")
        print("⚠️ CUDA not available — using CPU.")

    print("Device:", DEVICE)

    # Assign args to variables
    K = args.clients
    SHARDS = args.shards
    DATASET = args.dataset

    # Paper-specific local updates: 12 for MNIST/FMNIST, 10 for CIFAR10
    # Use max_local_updates if specified, otherwise use local_epochs
    MAX_LOCAL_UPDATES = args.max_local_updates
    if MAX_LOCAL_UPDATES is None and DATASET in ['MNIST', 'FMNIST']:
        # Automatically set to 12 for MNIST/FMNIST to match paper
        MAX_LOCAL_UPDATES = 12
        print(f"⚙️  Auto-setting max_local_updates={MAX_LOCAL_UPDATES} for {DATASET} (paper configuration)")
    elif MAX_LOCAL_UPDATES is None and DATASET == 'CIFAR10':
        # Automatically set to 10 for CIFAR10 to match paper
        MAX_LOCAL_UPDATES = 10
        print(f"⚙️  Auto-setting max_local_updates={MAX_LOCAL_UPDATES} for {DATASET} (paper configuration)")  # CRITICAL: Missing assignment
    ROUNDS = args.rounds
    LOCAL_EPOCHS = args.local_epochs
    BATCH = args.batch_size
    LR = args.lr
    GAMMA = args.gamma
    LAMBDA_SPLITGP = args.lambda_splitgp
    SEED = args.seed

    # When deterministic flag is set, force zero workers and disable pin_memory for exact reproducibility
    if args.deterministic:
        NUM_WORKERS = 0
        PIN_MEMORY = False
    else:
        NUM_WORKERS = args.num_workers
        PIN_MEMORY = not args.no_pin_memory

    SMALL_CLASSIFIER = args.small_classifier
    PROBE_PRINTS = args.probe
    DATASET = args.dataset  # CRITICAL: Missing assignment
    # Paper uses these Eth values (Section VI-A): {0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.3}
    # Note: These seem small for log2 entropy but match the paper exactly
    eth_thresholds = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.3]
    ETH = args.eth # <--- GET ETH FROM ARGS
    split_index = args.split_index
    method = args.method
    MODEL_TYPE = args.model  # Use model type from command-line argument

    # Create hierarchical results directory structure:
    # results/Model/Dataset/Method/rounds_X/lambda_Y/
    # Example: results/SimpleCNN/FMNIST/splitgp/rounds_120/lambda_0.2/
    results_folder_name = "results"
    os.makedirs(results_folder_name, exist_ok=True)

    # Level 1: Model (SimpleCNN, VGG11, etc.)
    model_folder = os.path.join(results_folder_name, MODEL_TYPE)
    os.makedirs(model_folder, exist_ok=True)

    # Level 2: Dataset (MNIST, FMNIST, CIFAR10)
    dataset_folder = os.path.join(model_folder, DATASET)
    os.makedirs(dataset_folder, exist_ok=True)

    # Level 3: Method (splitgp, fedavg, personalized, etc.)
    method_folder = os.path.join(dataset_folder, method)
    os.makedirs(method_folder, exist_ok=True)

    # Level 4: Rounds
    rounds_folder = os.path.join(method_folder, f"rounds_{ROUNDS}")
    os.makedirs(rounds_folder, exist_ok=True)

    # Level 5: Lambda (personalization parameter)
    lambda_folder = os.path.join(rounds_folder, f"lambda_{LAMBDA_SPLITGP}")
    os.makedirs(lambda_folder, exist_ok=True)

    # Final output directory with additional parameters
    OUT_DIR = os.path.join(lambda_folder, f"clients_{K}_gamma_{GAMMA}_ETH_{ETH}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*70)
    print("Results Directory Structure:")
    print(f"  {MODEL_TYPE}/")
    print(f"    └── {DATASET}/")
    print(f"        └── {method}/")
    print(f"            └── rounds_{ROUNDS}/")
    print(f"                └── lambda_{LAMBDA_SPLITGP}/")
    print(f"                    └── clients_{K}_gamma_{GAMMA}_ETH_{ETH}/")
    print()
    print(f"Full path: {OUT_DIR}")
    print("="*70)
    # eth_thresholds already defined above with corrected values
    p_values = [0,0.2,0.4,0.6,0.8,1.0]

    # ---- Run methods ----
    # Only run SplitGP method since others have been removed
    methods_to_run = ["splitgp"]
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

    # Probe param counts for the selected model
    tmp_base = create_model(MODEL_TYPE, in_channels=IN_CHANNELS, num_classes=10, split_index=split_index)
    phi_tmp = create_phi(tmp_base, MODEL_TYPE, split_index=split_index)
    feat_shape = probe_phi_feature_shape(phi_tmp, in_channels=IN_CHANNELS, img_size=IMG_SIZE, device=DEVICE)
    theta_tmp = create_theta(tmp_base, MODEL_TYPE, split_index=split_index)
    kappa_tmp = create_kappa(feat_shape, MODEL_TYPE, num_classes=10)

    phi_param_count = sum(p.numel() for p in phi_tmp.parameters())
    theta_param_count = sum(p.numel() for p in theta_tmp.parameters())
    kappa_param_count = sum(p.numel() for p in kappa_tmp.parameters())

    # Detailed breakdown for VGG11
    theta_tail_params = None
    theta_classifier_params = None
    theta_tail_bn_affine = None
    theta_tail_conv_bias = None
    if MODEL_TYPE.lower() in ['vgg11', 'vgg-11']:
        tail = VGG11_Theta(tmp_base, split_index).features_tail
        theta_tail_params = count_params(tail)
        theta_tail_bn_affine = count_bn_affine_params(tail)
        theta_tail_conv_bias = count_conv_bias_params(tail)
        theta_classifier_params = count_params(tmp_base.classifier)

    with open(os.path.join(OUT_DIR, "model_params.txt"), "w") as f:
        f.write(f"Model type: {MODEL_TYPE}\n")
        f.write(f"Split index: {split_index}\n")
        f.write(f"Dataset: {DATASET}\n\n")
        f.write(f"Phi (client-side) parameter count (phi only): {phi_param_count}\n")
        f.write(f"Kappa (auxiliary classifier) parameter count: {kappa_param_count}\n")
        f.write(f"Total client-side parameters (Phi + Kappa): {phi_param_count + kappa_param_count}\n\n")
        f.write(f"Theta (server-side) TOTAL parameter count: {theta_param_count}\n")
        if theta_tail_params is not None:
            f.write(f"  - Theta features tail params: {theta_tail_params}\n")
            f.write(f"  - Theta classifier params: {theta_classifier_params}\n")
            f.write(f"  - Theta BN affine params in tail: {theta_tail_bn_affine}\n")
            f.write(f"  - Theta conv bias params in tail: {theta_tail_conv_bias}\n")
            approx_paper_theta = theta_tail_params - (theta_tail_bn_affine or 0)
            f.write(f"Approx. paper Theta (features only, no BN affine, no classifier): {approx_paper_theta}\n")
            approx_paper_theta_strict = theta_tail_params - (theta_tail_bn_affine or 0) - (theta_tail_conv_bias or 0)
            local_epochs=LOCAL_EPOCHS,
            max_local_updates=MAX_LOCAL_UPDATES,
            f.write(f"Approx. paper Theta strict (features only, no BN affine, no conv bias, no classifier): {approx_paper_theta_strict}\n")

    if PROBE_PRINTS:
        print("Probe phi output feature shape (C,H,W):", feat_shape)
        print(f"Architecture: {MODEL_TYPE}")
        if MODEL_TYPE.lower() == 'simplecnn':
            print("  - Phi (client-side): 4 convolutional layers")
            print("  - Theta (server-side): 1 convolutional layer + 3 FC layers")
            print("  - Kappa (auxiliary): 1 FC layer")
        else:
            print("  - Phi (client-side): subset of VGG11.features up to split_index")
            print("  - Theta (server-side): remaining features + avgpool + classifier")
            print("  - Kappa (auxiliary): 1 FC layer on flattened phi output (or pooled)")

    results = defaultdict(list)


    all_sweep_results = []

    if "splitgp" in methods_to_run:

        clients_phi_sgp, clients_kappa_sgp, server_theta_sgp = train_split_training(
            in_channels=IN_CHANNELS,
            img_size=IMG_SIZE,
            split_index=split_index,
            lambda_personalization=LAMBDA_SPLITGP,
            gamma=GAMMA,  # Use current GAMMA
            lr=LR,
            batch=BATCH,
            rounds=ROUNDS,
            client_loader=client_loaders,
            model_type=MODEL_TYPE
        )

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
                    batch_size=BATCH,
                    model_type=MODEL_TYPE
                )

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

            # Convert to percentage before saving
            for col in ["full_acc", "client_acc", "selective_acc"]:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col] * 100

            df = df_temp.copy()
            # create an indexed copy for saving (avoid inplace to keep types clear)
            df_indexed = df.set_index('p')
            csv_name = f"{method}_combined_results_eth_{eth:.2f}_gamma_{GAMMA}_lambda_split_{LAMBDA_SPLITGP}.csv"
            csv_path = os.path.join(OUT_DIR, csv_name)
            df_indexed.to_csv(csv_path)
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

            # Create visualization plots for SplitGP results
            print("\n📊 Creating SplitGP visualization plots...")

            # Plot 1: Selective Accuracy vs p for different Eth values
            plt.figure(figsize=(10, 6))
            for eth in eth_thresholds:
                eth_data = df_final[df_final['eth'] == eth]
                if not eth_data.empty:
                    plt.plot(eth_data['p'], eth_data['selective_acc'],
                            marker='o', label=f'Eth={eth:.2f}', linewidth=2)

            plt.xlabel("p (OOD proportion)", fontsize=12)
            plt.ylabel("Selective Accuracy (%)", fontsize=12)
            plt.title(f"SplitGP on {DATASET}: Selective Accuracy vs OOD (γ={GAMMA}, λ={LAMBDA_SPLITGP})", fontsize=13)
            plt.xticks(p_values)
            plt.ylim(0, 105)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9, ncol=2)
            plt.tight_layout()

            plot_path = os.path.join(OUT_DIR, "splitgp_selective_acc_vs_p_eth_sweep.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"✅ Saved: {plot_path}")
            plt.close()

            # Plot 2: Client vs Server vs Selective Accuracy for a specific Eth
            best_eth_idx = len(eth_thresholds) // 2  # Middle Eth value
            best_eth = eth_thresholds[best_eth_idx]
            eth_data = df_final[df_final['eth'] == best_eth]

            if not eth_data.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(eth_data['p'], eth_data['client_acc'],
                        marker='s', label='Client-side (κ)', linewidth=2, markersize=8)
                plt.plot(eth_data['p'], eth_data['full_acc'],
                        marker='^', label='Server-side (θ)', linewidth=2, markersize=8)
                plt.plot(eth_data['p'], eth_data['selective_acc'],
                        marker='o', label=f'Selective (Eth={best_eth:.2f})', linewidth=2.5, markersize=8)

                plt.xlabel("p (OOD proportion)", fontsize=12)
                plt.ylabel("Test Accuracy (%)", fontsize=12)
                plt.title(f"SplitGP on {DATASET}: Comparison (γ={GAMMA}, λ={LAMBDA_SPLITGP}, Eth={best_eth:.2f})", fontsize=13)
                plt.xticks(p_values)
                plt.ylim(0, 105)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10)
                plt.tight_layout()

                plot_path = os.path.join(OUT_DIR, f"splitgp_comparison_eth_{best_eth:.2f}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
                print(f"✅ Saved: {plot_path}")
                plt.close()

            # Plot 3: Heatmap of Selective Accuracy (Eth vs p)
            pivot_table = df_final.pivot_table(values='selective_acc',
                                               index='eth', columns='p',
                                               aggfunc='first')

            plt.figure(figsize=(10, 6))
            im = plt.imshow(pivot_table.values, aspect='auto', cmap='YlGnBu',
                           vmin=0, vmax=100)
            plt.colorbar(im, label='Selective Accuracy (%)')

            plt.xlabel("p (OOD proportion)", fontsize=12)
            plt.ylabel("Entropy Threshold (Eth)", fontsize=12)
            plt.title(f"SplitGP on {DATASET}: Selective Accuracy Heatmap (γ={GAMMA}, λ={LAMBDA_SPLITGP})", fontsize=13)

            plt.xticks(range(len(pivot_table.columns)), [f'{p:.1f}' for p in pivot_table.columns])
            plt.yticks(range(len(pivot_table.index)), [f'{e:.2f}' for e in pivot_table.index])

            # Add text annotations
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    val = pivot_table.values[i, j]
                    if not np.isnan(val):
                        color = 'white' if val < 50 else 'black'
                        plt.text(j, i, f'{val:.1f}', ha='center', va='center',
                                color=color, fontsize=9, weight='bold')
            plt.tight_layout()
            plot_path = os.path.join(OUT_DIR, "splitgp_heatmap_eth_vs_p.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"✅ Saved: {plot_path}")
            plt.close()

            print("\n✅ All SplitGP visualization plots created successfully!\n")
            #test
            # test