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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ----------------------------
# Dataset loader helper
# ----------------------------
def get_datasets(name="MNIST"):
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
# ----------------------------
# Non-IID shard partition
# ----------------------------
def create_clients_shards(dataset, K=20, shards=100):
    n = len(dataset)
    idxs = np.arange(n)
    labels = np.array(dataset.targets)
    # sort indices by label
    sorted_idx = idxs[np.argsort(labels)]
    shard_size = n // shards
    shard_idxs = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(shards)]
    random.shuffle(shard_idxs)
    clients = {}
    for i in tqdm(range(K),desc="Creating clients"):
        # each client gets 2 shards
        clients[i] = np.concatenate(shard_idxs[2 * i:2 * i + 2])
    return clients

# ----------------------------
# Simple CNN (for MNIST/FM) - accepts in_channels
# You can replace with VGG for CIFAR10 later.
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Kappa(nn.Module):
    """Auxiliary classifier mapping flattened phi features -> class logits."""
    def __init__(self, feature_shape, num_classes=10):
        super().__init__()
        flat = int(np.prod(feature_shape))
        self.fc = nn.Linear(flat, num_classes)

    def forward(self, h):
        return self.fc(torch.flatten(h, 1))

def probe_phi_feature_shape(phi_module, in_channels=1, img_size=28, device='cpu'):
    """Run a dummy forward to get the (C,H,W) feature shape produced by phi."""
    phi_module = phi_module.to(device)
    phi_module.eval()
    with torch.no_grad():
        sample = torch.randn(1, in_channels, img_size, img_size).to(device)
        feat = phi_module(sample)
        return tuple(feat.shape[1:])  # (C,H,W)

class Phi(nn.Module):
    """
    Client-side feature extractor: reuse first conv blocks of SimpleCNN.
    We will instantiate Phi from a 'base' SimpleCNN instance to reuse conv layers.
    """
    def __init__(self, base: SimpleCNN):
        super().__init__()
        # first four conv blocks from the base network
        self.net = nn.Sequential(
            base.conv1, nn.ReLU(),
            base.pool,
            base.conv2, nn.ReLU(),
            base.pool,
            base.conv3, nn.ReLU(),
            base.conv4, nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
# ----------------------------
# DataLoader helpers
# ----------------------------


def train_phi_kappa_local(batch=None, lr=None):
    """
    Train phi + kappa locally for each client (no server).
    Returns list of state_dicts for phi and list for kappa (CPU).
    """
    # 1) Build base model and phi template
    base = SimpleCNN(in_channels=IN_CHANNELS).to(device)
    phi_template = Phi(base).to(device)

    # 2) probe feature shape so we can create Kappa
    feat_shape = probe_phi_feature_shape(Phi(base), in_channels=IN_CHANNELS, img_size=IMG_SIZE, device=device)

    # 3) create client phi and kappa modules
    clients_phi = [deepcopy(phi_template).to(device) for _ in range(K)]
    clients_kappa = [Kappa(feat_shape, num_classes=10).to(device) for _ in range(K)]

    # Initialize all kappa the same (optional)
    # common_kappa_state = deepcopy(clients_kappa[0].state_dict())

    phi_states_cpu = []
    kappa_states_cpu = []

    for k in tqdm(range(K), desc="Training Clients (phi+kappa training)"):
        phi = clients_phi[k]
        kappa = clients_kappa[k]
        # initialize from template (they already are)
        # set up optimizer for phi + kappa
        opt = optim.SGD(list(phi.parameters()) + list(kappa.parameters()), lr=lr)

        loader = get_client_loader(k, batch=batch)
        phi.train()
        kappa.train()
        for r in tqdm(range(ROUNDS),desc=f"Client {k} rounds", leave=False):
            for _ in tqdm(range(LOCAL_EPOCHS),desc=f"Client {k} local epochs", leave=False):
                for xb, yb in loader:
                    # ensure tensors
                    xb = xb.to(device)
                    # some dataset variants may return ints for yb if batch_size=1
                    if not torch.is_tensor(yb):
                        yb = torch.tensor(yb, dtype=torch.long)
                    yb = yb.to(device)

                    h = phi(xb)
                    out = kappa(h)
                    loss = F.cross_entropy(out, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        # after local training, save CPU copies of state_dicts
        phi_states_cpu.append({name: param.cpu().clone() for name, param in phi.state_dict().items()})
        kappa_states_cpu.append({name: param.cpu().clone() for name, param in kappa.state_dict().items()})
        # free GPU memory for next client
        del phi, kappa, opt
        torch.cuda.empty_cache()

    return phi_states_cpu, kappa_states_cpu
def evaluate_phi_kappa_local(phi_states,
                             kappa_states,
                             per_client_testsets,
                             batch_size=None):
    """
    Evaluate local phi+kappa for each client, return average accuracy.
    phi_states, kappa_states: lists of cpu state_dicts (length K)
    per_client_testsets: list of Subset (length K)
    """
    accs = []
    for k in tqdm(range(K),desc="Evaluating phi+kappa models"):
        # rebuild phi and kappa on device, load state
        base = SimpleCNN(in_channels=IN_CHANNELS).to(device)
        phi = Phi(base).to(device)
        feat_shape = probe_phi_feature_shape(Phi(base), in_channels=IN_CHANNELS, img_size=IMG_SIZE, device=device)
        kappa = Kappa(feat_shape).to(device)

        phi.load_state_dict(phi_states[k])
        kappa.load_state_dict(kappa_states[k])
        phi.eval()
        kappa.eval()

        correct = 0
        total = 0
        loader = DataLoader(per_client_testsets[k], batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                if not torch.is_tensor(yb):
                    yb = torch.tensor(yb, dtype=torch.long)
                yb = yb.to(device)
                h = phi(xb)
                out = kappa(h)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accs.append(correct/total if total>0 else 0.0)
        del phi, kappa
        torch.cuda.empty_cache()

    return float(np.mean(accs))
def get_client_loader(client_idx, batch=None):
    idxs = clients_indices[client_idx]
    subset = Subset(trainset, idxs)
    return DataLoader(subset, batch_size=batch, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def client_test_set_for_p(client_idx, p):
    """
    Build local test set for client: all test samples of client's main classes
    + p * (#main) random out-of-distribution samples from test set.
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

# ----------------------------
# Train Personalized local-only models
# ----------------------------
def train_personalized_local_only(batch=None):
    """
    Each client trains a full model locally from the same initialization, no aggregation.
    Returns: list of client state_dicts (CPU tensors) to save GPU memory.
    """
    base = SimpleCNN(in_channels=IN_CHANNELS).to(device)
    init_state = deepcopy(base.state_dict())
    client_states = []

    for k in tqdm(range(K), desc="Clients (personalized training)"):
        # new model per client initialized to same weights
        model = SimpleCNN(in_channels=IN_CHANNELS).to(device)
        model.load_state_dict(init_state)
        opt = optim.SGD(model.parameters(), lr=LR)
        loader = get_client_loader(k,batch=batch)
        model.train()

        for r in tqdm(range(ROUNDS), desc=f"Client {k} rounds", leave=False):
            for _ in tqdm(range(LOCAL_EPOCHS),desc=f"Client {k} local epochs", leave=False):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        # move state to CPU to save GPU memory
        state_cpu = {kname: v.cpu() for kname, v in model.state_dict().items()}
        client_states.append(state_cpu)
        # free GPU memory for next client
        del model
        torch.cuda.empty_cache()
    return client_states

# ----------------------------
# Evaluation of full models
# ----------------------------
def evaluate_method_full_models(clients_state_dicts, per_client_testsets,batch_size=None):
    """
    clients_state_dicts: list of state_dicts (CPU)
    per_client_testsets: list of Subset datasets (length K)
    returns average accuracy across clients
    """
    accs = []
    for k in tqdm(range(K),desc="Evaluating personalized models"):
        state = clients_state_dicts[k]
        model = SimpleCNN(in_channels=IN_CHANNELS).to(device)
        model.load_state_dict(state)
        model.eval()
        correct = 0
        total = 0
        loader = DataLoader(per_client_testsets[k], batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accs.append(correct / total if total > 0 else 0.0)
        # free GPU memory
        del model
        torch.cuda.empty_cache()
    return float(np.mean(accs))
# ----------------------------
# Energy estimation and Eth logic
# ----------------------------
def estimate_energy(x):
    """Dummy energy estimation for local inference. Replace with real model if available."""
    # For demonstration, use input size as proxy for energy
    return x.numel() * 1e-6  # scale factor for illustration

def evaluate_phi_kappa_with_eth(phi_states, kappa_states, per_client_testsets, batch_size, Eth):
    """
    Evaluate phi+kappa for each client, using Eth to decide local prediction or server offload.
    Returns: avg local accuracy, portion of samples predicted locally.
    """
    local_accs = []
    local_counts = []
    total_counts = []
    for k in range(K):
        base = SimpleCNN(in_channels=IN_CHANNELS).to(device)
        phi = Phi(base).to(device)
        feat_shape = probe_phi_feature_shape(Phi(base), in_channels=IN_CHANNELS, img_size=IMG_SIZE, device=device)
        kappa = Kappa(feat_shape).to(device)
        phi.load_state_dict(phi_states[k])
        kappa.load_state_dict(kappa_states[k])
        phi.eval(); kappa.eval()
        correct_local = 0
        total_local = 0
        total_samples = 0
        loader = DataLoader(per_client_testsets[k], batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                energy = estimate_energy(xb)
                total_samples += yb.size(0)
                if energy < Eth:
                    h = phi(xb)
                    out = kappa(h)
                    pred = out.argmax(dim=1)
                    correct_local += (pred == yb).sum().item()
                    total_local += yb.size(0)
                # else: offload to server (not implemented, so skip)
        local_accs.append(correct_local / total_local if total_local > 0 else 0.0)
        local_counts.append(total_local)
        total_counts.append(total_samples)
        del phi, kappa
        torch.cuda.empty_cache()
    avg_local_acc = float(np.mean(local_accs))
    avg_local_portion = float(np.sum(local_counts)) / float(np.sum(total_counts)) if np.sum(total_counts) > 0 else 0.0
    return avg_local_acc, avg_local_portion

if __name__=='__main__':
    # ----------------------------
    # Config (change these)
    # ----------------------------
    DATASET = "CIFAR10"  # options: "MNIST", "FMNIST", "CIFAR10"
    K =2  # number of clients (paper uses 50)
    SHARDS = 100  # number of shards (paper uses 100)
    ROUNDS =1  # global rounds (paper: 120 for MNIST; increase for better accuracy)
    LOCAL_EPOCHS = 1
    BATCH = 50
    LR = 0.01
    SEED = 42

    NUM_WORKERS = 4
    PIN_MEMORY = True

    # p values to evaluate (relative portion of OOD vs main)
    p_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(f"Configration : Rounds : {ROUNDS}")
    # output files
    OUT_DIR = f"Phi+Kappa_(local-only)_{DATASET}"
    os.makedirs(OUT_DIR, exist_ok=True)

    trainset, testset, IN_CHANNELS, IMG_SIZE = get_datasets(DATASET)
    train_labels = np.array(trainset.targets)
    test_labels = np.array(testset.targets)

    clients_indices = create_clients_shards(trainset, K=K, shards=SHARDS)
    print(f"Created {K} clients; approx samples per client = {len(clients_indices[0])}")

    # Precompute per-client test sets for each p
    per_p_testsets = {}
    for p in p_values:
        print("Building test sets for p =", p)
        per_p_testsets[p] = [client_test_set_for_p(k, p) for k in range(K)]

    print("Training phi + kappa locally (no server)...")
    phi_states, kappa_states = train_phi_kappa_local(batch=BATCH, lr=LR)

    results_phi_kappa = []
    for p in p_values:
        avg_acc = evaluate_phi_kappa_local(phi_states, kappa_states, per_p_testsets[p], batch_size=BATCH)
        print(f"p={p:.2f} phi+kappa avg acc: {avg_acc:.4f}")
        results_phi_kappa.append(avg_acc)

    df = pd.DataFrame({'Phi+Kappa (local-only)': results_phi_kappa}, index=p_values)
    df.index.name = 'p'
    df.to_csv(os.path.join(OUT_DIR, 'phi_kappa_results.csv'))
    # plot ...
    print(list(df))
        # Train personalized models
    print("Training personalized (local-only) models ...")

    # personal_states = train_personalized_local_only(batch=BATCH)

    # Evaluate for each p
    # results = {}
    # for p in p_values:
    #     avg_acc = evaluate_method_full_models(personal_states, per_p_testsets[p],batch_size=BATCH)
    #     results.setdefault("Personalized", []).append(avg_acc)
    #     print(f"p={p:.2f} Personalized avg acc: {avg_acc:.4f}")

    # # Save results and plot
    # df = pd.DataFrame(results, index=p_values)
    # df.index.name = "p"
    # csv_path = os.path.join(OUT_DIR, f"personalized_results.csv")
    # df.to_csv(csv_path)
    # print("Saved CSV ->", csv_path)

    # quick plot
    plt.figure(figsize=(6, 4))
    plt.plot(p_values, df['Phi+Kappa (local-only)'], marker='o', label="Phi+Kappa (local-only)")
    plt.xlabel("p (relative portion of OOD samples)")
    plt.ylabel("Avg client test accuracy")
    plt.title("Phi+Kappa (local-only): Accuracy vs p")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phi_kappa_accuracy_vs_p.png"), dpi=200)
    plt.show()

    Eth_values = [0.01, 0.02, 0.05, 0.1, 0.2]  # Example Eth thresholds
    eth_results = {'Eth': [], 'Local Accuracy': [], 'Local Portion': []}
    for Eth in Eth_values:
        avg_acc, avg_portion = evaluate_phi_kappa_with_eth(phi_states, kappa_states, per_p_testsets[0.0], batch_size=BATCH, Eth=Eth)
        print(f"Eth={Eth:.3f}: Local acc={avg_acc:.4f}, Local portion={avg_portion:.2f}")
        eth_results['Eth'].append(Eth)
        eth_results['Local Accuracy'].append(avg_acc)
        eth_results['Local Portion'].append(avg_portion)
    df_eth = pd.DataFrame(eth_results)
    df_eth.to_csv(os.path.join(OUT_DIR, 'phi_kappa_eth_results.csv'))
    plt.figure(figsize=(6,4))
    plt.plot(df_eth['Eth'], df_eth['Local Accuracy'], marker='o', label='Local Accuracy')
    plt.plot(df_eth['Eth'], df_eth['Local Portion'], marker='x', label='Local Portion')
    plt.xlabel('Eth (Energy Threshold)')
    plt.ylabel('Value')
    plt.title('Effect of Eth on Local Accuracy and Portion')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'phi_kappa_eth_plot.png'), dpi=200)
    plt.show()
