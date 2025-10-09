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
    for i in range(K):
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

# ----------------------------
# DataLoader helpers
# ----------------------------
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
            for _ in range(LOCAL_EPOCHS):
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
    for k in range(K):
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
if __name__=='__main__':
    # ----------------------------
    # Config (change these)
    # ----------------------------
    DATASET = "CIFAR10"  # options: "MNIST", "FMNIST", "CIFAR10"
    K = 50  # number of clients (paper uses 50)
    SHARDS = 100  # number of shards (paper uses 100)
    ROUNDS = 800  # global rounds (paper: 120 for MNIST; increase for better accuracy)
    LOCAL_EPOCHS = 1
    BATCH = 50
    LR = 0.01
    SEED = 42

    NUM_WORKERS = 4
    PIN_MEMORY = True

    # p values to evaluate (relative portion of OOD vs main)
    p_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # output files
    OUT_DIR = f"results_personalized_{DATASET}"
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

        # Train personalized models
    print("Training personalized (local-only) models ...")
    personal_states = train_personalized_local_only(batch=BATCH)

    # Evaluate for each p
    results = {}
    for p in p_values:
        avg_acc = evaluate_method_full_models(personal_states, per_p_testsets[p],batch_size=BATCH)
        results.setdefault("Personalized", []).append(avg_acc)
        print(f"p={p:.2f} Personalized avg acc: {avg_acc:.4f}")

    # Save results and plot
    df = pd.DataFrame(results, index=p_values)
    df.index.name = "p"
    csv_path = os.path.join(OUT_DIR, f"personalized_results.csv")
    df.to_csv(csv_path)
    print("Saved CSV ->", csv_path)

    # quick plot
    plt.figure(figsize=(6, 4))
    plt.plot(p_values, results["Personalized"], marker='o', label="Personalized")
    plt.xlabel("p (relative portion of OOD samples)")
    plt.ylabel("Avg client test accuracy")
    plt.title("Personalized FL: Accuracy vs p")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "personalized_accuracy_vs_p.png"), dpi=200)
    plt.show()