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
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory in DataLoader")
    parser.add_argument("--eth", type=float, default=0.5, help="Entropy threshold (Eth) for selective offloading") # <--- ADD THIS LINE

    # Model toggles
    parser.add_argument("--small-classifier", action="store_true", help="Use smaller FC layers (avoid OOM)")
    parser.add_argument("--probe", action="store_true", help="Print probe feature shapes")

    # Method choice
    parser.add_argument("--method", type=str, default="splitgp",
                        choices=["splitgp", "multi-exit", "personalized", "fedavg", "all"],
                        help="Which training method to run")
    parser.add_argument("--split_index", type=int, default=11, help="SplitIndex")
    parser.add_argument("--model", type=str, default="SimpleCNN", help="Model type for results folder (e.g., SimpleCNN, VGG11)")

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
    shard_idxs = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(shards)]
    random.shuffle(shard_idxs)
    clients = {}
    for i in range(K):
        clients[i] = np.concatenate(shard_idxs[2 * i:2 * i + 2])
    return clients


class SimpleCNN(nn.Module):
    """CNN model for MNIST/FMNIST with 5 conv layers + 3 FC layers (Paper-aligned)"""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # Convolutions (all 3x3, padding=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding=1)   # φ ends here
        self.conv5 = nn.Conv2d(256, 256, 3, 1, padding=1)   # θ starts here
        self.pool = nn.MaxPool2d(2)

        # Compute fc1 input size depending on dataset family
        # MNIST/FMNISt (in_channels==1): φ output 28->14->7->3 => 256*3*3
        # CIFAR10 (in_channels==3):      32->16->8->4        => 256*4*4
        fc1_in = 256 * (3 * 3 if in_channels == 1 else 4 * 4)

        # Fully connected layers (θ side): 2304/4096 -> 1024 -> 512 -> 10
        self.fc1 = nn.Linear(fc1_in, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Full-model forward (used only for baselines)
        x = F.relu(self.conv1(x)); x = self.pool(x)     # 28->14 (or 32->16)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)); x = self.pool(x)     # 14->7 (or 16->8)
        x = F.relu(self.conv4(x)); x = self.pool(x)     # 7->3 (or 8->4)
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Phi(nn.Module):
    """Client-side feature extractor: conv1..conv4 with pooling after conv1, conv3, conv4."""
    def __init__(self, cnn_model):
        super().__init__()
        self.conv1 = cnn_model.conv1
        self.conv2 = cnn_model.conv2
        self.conv3 = cnn_model.conv3
        self.conv4 = cnn_model.conv4
        self.pool = cnn_model.pool

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)     # 28->14
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)); x = self.pool(x)     # 14->7
        x = F.relu(self.conv4(x)); x = self.pool(x)     # 7->3
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
    n_ood = int(round(p * n_main))
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
        cnn = SimpleCNN(in_channels=in_channels, num_classes=10).to(DEVICE)
        cnn.load_state_dict(state)
        cnn.eval()
        correct = 0
        total = 0
        loader = DataLoader(per_client_testsets[k], batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                if not torch.is_tensor(yb):
                    yb = torch.tensor(yb, dtype=torch.long)
                yb = yb.to(DEVICE)
                out = cnn(xb)
                pred = out.argmax(dim=1)

                # Fix for correct += (pred == yb).sum().item()
                if not torch.is_tensor(pred):
                    pred = torch.tensor(pred)
                if not torch.is_tensor(yb):
                    yb = torch.tensor(yb)
                correct += (pred == yb).sum().item()

                total += yb.size(0)
        accs.append(correct / total if total > 0 else 0.0)
        del cnn
        torch.cuda.empty_cache()
    return float(np.mean(accs))





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
    base = SimpleCNN(in_channels=in_channels, num_classes=10).to(DEVICE)
    server_theta = Theta(base).to(DEVICE)
    server_theta.load_state_dict({k: v.to(DEVICE) for k, v in server_theta_state.items()})
    server_theta.eval()

    full_accs = []
    client_accs = []
    selective_accs = []  # <--- MODIFIED TO INCLUDE SELECTIVE

    for k in range(K):
        # ... (rebuild phi & kappa - same as before)
        base_local = SimpleCNN(in_channels=in_channels, num_classes=10).to(DEVICE)
        phi = Phi(base_local).to(DEVICE)
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
                # Fix for correct_client += (pred_c == yb).sum().item()
                if not torch.is_tensor(pred_c):
                    pred_c = torch.tensor(pred_c)
                if not torch.is_tensor(yb):
                    yb = torch.tensor(yb)
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
    base = SimpleCNN(in_channels=in_channels, num_classes=10).to(DEVICE)
    phi_template = Phi(base).to(DEVICE)
    theta_template = Theta(base).to(DEVICE)

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

        for k in tqdm(range(K),desc=" Clients Number ", leave=False):
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
        SimpleCNN(in_channels=in_channels, num_classes=10).to(DEVICE)
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
    global_model = SimpleCNN(in_channels=in_channels, num_classes=10).to(DEVICE)
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
    SMALL_CLASSIFIER = False
    PROBE_PRINTS = args.probe
    DATASET = args.dataset
    ETH = args.eth # <--- GET ETH FROM ARGS
    split_index = args.split_index
    method = args.method
    MODEL_TYPE = args.model  # Use model type from command-line argument
    results_folder_name = "results"
    os.makedirs(results_folder_name, exist_ok=True)
    model_folder = os.path.join(results_folder_name, MODEL_TYPE)
    os.makedirs(model_folder, exist_ok=True)
    dataset_folder = os.path.join(model_folder, DATASET)
    os.makedirs(dataset_folder, exist_ok=True)
    OUT_DIR = os.path.join(dataset_folder, f"splitgp_method_{method}_rounds_{ROUNDS}_clients_{K}_gamma_{GAMMA}_lambda_split_{LAMBDA_SPLITGP}_ETH_{ETH}")
    print("out dir", OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    eth_thresholds = [0.05,0.1,0.2,0.4,0.8,1.2,1.6,2.3]
    p_values = [0,0.2,0.4,0.6,0.8,1.0]

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

    # CNN model doesn't need split_index like VGG11 - phi is fixed to first 4 conv layers
    tmp_base = SimpleCNN(in_channels=IN_CHANNELS, num_classes=10)

    # Optional: probe shapes
    phi_tmp = Phi(tmp_base)
    feat_shape = probe_phi_feature_shape(phi_tmp, in_channels=IN_CHANNELS, img_size=IMG_SIZE, device=DEVICE)
    theta_tmp = Theta(tmp_base)
    kappa_tmp = Kappa(feat_shape, num_classes=10)
    phi_param_count = sum(p.numel() for p in phi_tmp.parameters())
    theta_param_count = sum(p.numel() for p in theta_tmp.parameters())
    kappa_param_count = sum(p.numel() for p in kappa_tmp.parameters())
    with open(os.path.join(OUT_DIR, "model_params.txt"), "w") as f:
        f.write(f"Phi (client-side) parameter count: {phi_param_count}\n")
        f.write(f"Theta (server-side) parameter count: {theta_param_count}\n")
        f.write(f"Kappa (auxiliary classifier) parameter count: {kappa_param_count}\n")
        f.write(f"\nTotal client-side parameters (Phi + Kappa): {phi_param_count + kappa_param_count}\n")
        f.write(f"Total server-side parameters (Theta): {theta_param_count}\n")

    if PROBE_PRINTS:
        print("Probe phi output feature shape (C,H,W):", feat_shape)
        print("CNN Architecture:")
        print("  - Phi (client-side): 4 convolutional layers")
        print("  - Theta (server-side): 1 convolutional layer + 3 FC layers")
        print("  - Kappa (auxiliary): 1 FC layer")

    results = defaultdict(list)

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

        for p in p_values:
            full_acc, client_acc, selective_acc = evaluate_method_split(clients_phi_me,
                                                         clients_kappa_me,
                                                         server_theta_me,
                                                         per_p_testsets[p],
                                                         in_channels=IN_CHANNELS,
                                                         img_size=IMG_SIZE,
                                                         split_index=split_index,
                                                         threshold=ETH,
                                                         batch_size=BATCH)
            results['Multi-Exit'].append(full_acc)
            # results['Multi-Exit-Clien'].append(client_acc)

            print(f"p={p:.2f} Multi-Exit full acc: {full_acc:.4f}  client acc: {client_acc:.4f}, selective_acc : {selective_acc:.4f} ")

        df = pd.DataFrame(results, index=p_values)
        df.index.name = 'p'
        csv_path = os.path.join(OUT_DIR, "multi-exit_combined_results_eth_{ETH}_gamma_{GAMMA}_lambda_split_{LAMBDA_SPLITGP}.csv")
        df.to_csv(csv_path)
        print("Saved CSV ->", csv_path)

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

            # Convert to percentage before saving
            for col in ["full_acc", "client_acc", "selective_acc"]:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col] * 100

            df = df_temp.copy()
            df = df.set_index('p')
            csv_name = f"{method}_combined_results_eth_{eth:.2f}_gamma_{GAMMA}_lambda_split_{LAMBDA_SPLITGP}.csv"
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
            os.path.join(OUT_DIR, "personalized_sweep.csv"))

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


    # Before saving any DataFrame to CSV, convert accuracy columns to percentage
    accuracy_cols = ["selective_acc", "full_acc", "client_acc", "Personalized", "FedAvg Global"]

    def save_df_to_csv(df, csv_path):
        for col in accuracy_cols:
            if col in df.columns:
                df[col] = df[col] * 100
        df.to_csv(csv_path, index=False)

    # Plot
    # Plot results only if there are any methods in the results dict
    if results:
        print("Plotting results ...")
        plt.figure(figsize=(10, 6))
        for method_name, accs in results.items():
            # Convert to percentage for plotting
            accs_pct = [a * 100 for a in accs]
            plt.plot(p_values, accs_pct, marker='o', label=method_name, linewidth=2)

        plt.xlabel("p (OOD proportion)", fontsize=12)
        plt.ylabel("Test Accuracy (%)", fontsize=12)
        plt.title(f"{DATASET} - Accuracy vs OOD Proportion (p)", fontsize=14)
        plt.xticks(p_values)
        plt.ylim(0, 105)  # Set y-axis to 0-105% for better visualization
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        plot_png = os.path.join(OUT_DIR, "accuracy_vs_p.png")
        plot_pdf = os.path.join(OUT_DIR, "accuracy_vs_p.pdf")
        plt.savefig(plot_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_pdf, bbox_inches='tight')
        print(f"Saved plot -> {plot_png}")
        plt.close()
    else:
        print("No results to plot (results dict is empty).")
