"""
splitgp_compare_mnist.py

Compare Personalized FL, Generalized FL (FedAvg), Multi-Exit (lambda=0), and SplitGP (lambda=0.2)
on MNIST with shard-based non-IID partitioning. Produces Accuracy vs p plot.

Note: This is a simplified reproduction for research/dev. You can scale K/rounds to match paper.
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import os


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

K = 50  # number of clients (set to 50 for paper-scale; reduced here for speed)
SHARDS = 100  # number of shards (paper used 100)
ROUNDS = 800  # global rounds (paper: 120 for MNIST) -> increase to 120 for full replicate
LOCAL_EPOCHS = 1  # epochs per round per client (paper uses 1 epoch per round)
BATCH = 50  # mini-batch as in paper
LR = 0.01
GAMMA = 0.5  # multi-exit weight in objective (paper commonly uses 0.5)
LAMBDA_SPLITGP = 0.2  # personalization weight for SplitGP
SEED = 42

p_values = [0.0, 0.2, 0.4, 0.6, 0.8,1]  # values requested

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



# -----------------------------
# Non-IID shard partition
# -----------------------------
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
    for i in tqdm(range(K),desc="create_clients_shards"):
        # 2 shards per client
        clients[i] = np.concatenate(shard_idxs[2 * i:2 * i + 2])
    return clients


# -----------------------------
# Model definitions (CNN splitable)
# -----------------------------
class BaseCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # Conv blocks
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # FCs
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Split into phi (first 4 convs) and theta (remaining conv+FC)
class Phi(nn.Module):
    def __init__(self, base: BaseCNN):
        super().__init__()
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


class Theta(nn.Module):
    def __init__(self, base: BaseCNN):
        super().__init__()
        self.net = nn.Sequential(
            base.conv5, nn.ReLU(),
            base.gap,
            nn.Flatten(),
            base.fc1, nn.ReLU(),
            base.fc2, nn.ReLU(),
            base.fc3
        )

    def forward(self, h):
        return self.net(h)


# Kappa will be built dynamically once phi output shape known
class Kappa(nn.Module):
    def __init__(self, feature_shape, num_classes=10):
        super().__init__()
        flat = int(np.prod(feature_shape))
        self.fc = nn.Linear(flat, num_classes)

    def forward(self, h):
        return self.fc(torch.flatten(h, 1))


# Helper to get feature shape of phi for MNIST input
# Helper to get feature shape of phi for input
def probe_phi_feature_shape(phi_module, in_channels=1, img_size=28):
    with torch.no_grad():
        sample = torch.randn(1, in_channels, img_size, img_size).to(next(phi_module.parameters()).device)
        feat = phi_module(sample)
        return tuple(feat.shape[1:])



# -----------------------------
# Utilities: DataLoaders, FedAvg, evaluation
# -----------------------------
def get_client_loader(client_idx, batch=BATCH):
    idxs = clients_indices[client_idx]
    subset = Subset(trainset, idxs)
    return DataLoader(subset, batch_size=batch, shuffle=True)


def client_test_set_for_p(client_idx, p):
    """
    Build local test set for client: all test samples of client's main classes
    + p * (#main) random out-of-distribution samples from test set.
    """
    train_idxs = clients_indices[client_idx]
    classes, _ = np.unique(np.array(trainset.targets)[train_idxs], return_counts=True)
    main_mask = np.isin(test_labels, classes)
    main_idxs = np.where(main_mask)[0]
    n_main = len(main_idxs)
    n_ood = int(round(p * n_main))
    # sample ood indices from classes not in main
    non_main_idxs = np.where(~main_mask)[0]
    if n_ood > 0:
        ood_sample = np.random.choice(non_main_idxs, n_ood, replace=False)
        final_idxs = np.concatenate([main_idxs, ood_sample])
    else:
        final_idxs = main_idxs
    return Subset(testset, final_idxs)


def evaluate_method_full_models(clients_models, test_sets):
    """
    clients_models: list of full-models per client (nn.Module) OR a single global model
    test_sets: list of DataLoaders for each client
    Return average accuracy across clients
    """
    accs = []
    for k, test_loader in tqdm(enumerate(test_sets),desc="evaluate_method_full_models"):
        if isinstance(clients_models, list):
            model = clients_models[k]
        else:
            model = clients_models  # global model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in DataLoader(test_loader, batch_size=256):
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accs.append(correct / total if total > 0 else 0.0)
    return np.mean(accs)


def evaluate_method_split(clients_phi, clients_kappa, server_theta, test_sets):
    """Evaluate split model deployed: client uses phi+kappa unless offloaded.
       For evaluation here we compute full-model outputs by feeding phi->theta to measure full-model accuracy as well as client-side.
       But for aggregated test accuracy we will use full-model (phi+theta) to simulate final deployed SplitGP decision with Eth large enough.
       For simplicity, we compute full-model accuracy by running phi->theta and client accuracy phi->kappa.
    """
    server_theta.eval()
    client_full_accs = []
    client_client_accs = []
    for k, test_set in enumerate(test_sets):
        phi = clients_phi[k];
        kappa = clients_kappa[k]
        phi.eval();
        kappa.eval()
        correct_full = 0;
        correct_client = 0;
        total = 0
        with torch.no_grad():
            for xb, yb in DataLoader(test_set, batch_size=256):
                xb, yb = xb.to(device), yb.to(device)
                h = phi(xb)
                out_c = kappa(h).argmax(1)
                out_s = server_theta(h).argmax(1)
                # full: server output; client: client output
                correct_full += (out_s == yb).sum().item()
                correct_client += (out_c == yb).sum().item()
                total += yb.size(0)
        client_full_accs.append(correct_full / total if total > 0 else 0.0)
        client_client_accs.append(correct_client / total if total > 0 else 0.0)
    # return average full-model accuracy (what matters for generalization) and client accuracy (personalization)
    return np.mean(client_full_accs), np.mean(client_client_accs)


# -----------------------------
# Train functions for each method
# -----------------------------
def train_personalized_local_only():
    """
    Each client trains a full model locally from the same initialization, no aggregation.
    Return list of trained models (one per client).
    """
    base_model = BaseCNN().to(device)
    init_state = deepcopy(base_model.state_dict())
    client_models = []
    for k in tqdm(range(K),desc="train_personalized_local_only"):
        model = BaseCNN().to(device)
        model.load_state_dict(init_state)
        opt = optim.SGD(model.parameters(), lr=LR)
        loader = get_client_loader(k)
        model.train()
        for r in tqdm(range(ROUNDS),desc=f"(train_personalized_local_only) Client {k} rounds", leave=False):
            for _ in tqdm(range(LOCAL_EPOCHS),desc=f"Client {k} local epochs", leave=False):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        client_models.append(model)
    return client_models


def train_fedavg_global():
    """
    Standard FedAvg training of full model (global). Return global model.
    """
    global_model = BaseCNN().to(device)
    global_state = deepcopy(global_model.state_dict())

    for r in tqdm(range(ROUNDS),desc=" Rounds train_fedavg_global"):
        client_states = []
        for k in tqdm(range(K),desc=" Clients train_fedavg_global"):
            model = BaseCNN().to(device)
            model.load_state_dict(global_state)
            opt = optim.SGD(model.parameters(), lr=LR)
            loader = get_client_loader(k)
            model.train()
            for _ in tqdm(range(LOCAL_EPOCHS),desc=f"(train_fedavg_global) Client {k} local epochs", leave=False):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            client_states.append(deepcopy(model.state_dict()))
        # aggregate by simple average
        new_state = {}
        for k_key in client_states[0].keys():
            new_state[k_key] = sum(client_states[i][k_key] for i in range(K)) / K
        global_state = new_state
        global_model.load_state_dict(global_state)
    return global_model


def train_split_training(lambda_personalization=0.2):
    """
    Split training (SplitGP family). Returns:
      - clients_phi: list of phi modules (per client)
      - clients_kappa: list of kappa modules (per client)
      - server_theta: single theta module
    lambda_personalization: lambda in Eq.(10); lambda=0 -> no personalization (multi-exit baseline)
    """

    base = BaseCNN(in_channels=IN_CHANNELS, num_classes=10)
    phi_template = Phi(base).to(device)
    theta_template = Theta(base).to(device)
    feat_shape = probe_phi_feature_shape(Phi(base), in_channels=IN_CHANNELS, img_size=IMG_SIZE)
    # kappa template per client -> dimension = flattened phi output
    # initialize models
    clients_phi = [deepcopy(phi_template).to(device) for _ in range(K)]
    clients_kappa = [Kappa(feat_shape).to(device) for _ in range(K)]
    server_theta = deepcopy(theta_template).to(device)

    # set initial states
    phi_state = deepcopy(phi_template.state_dict())
    kappa_state = deepcopy(clients_kappa[0].state_dict())
    theta_state = deepcopy(theta_template.state_dict())

    for r in tqdm(range(ROUNDS),desc=" Rounds train_split_training"):
        # per-client local updates
        client_phi_states = []
        client_kappa_states = []
        client_theta_states = []
        for k in tqdm(range(K),desc=" Clients train_split_training",leave=False):
            # load current states
            clients_phi[k].load_state_dict(phi_state)
            clients_kappa[k].load_state_dict(kappa_state)
            server_theta.load_state_dict(theta_state)

            # local optimizer over phi,kappa,theta (theta here is local replica used for computing server-side loss)
            opt = optim.SGD(list(clients_phi[k].parameters()) + list(clients_kappa[k].parameters()) + list(
                server_theta.parameters()), lr=LR)
            loader = get_client_loader(k)
            clients_phi[k].train()
            clients_kappa[k].train()
            server_theta.train()
            for _ in tqdm(range(LOCAL_EPOCHS),desc=f"Client {k} local epochs", leave=False):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    h = clients_phi[k](xb)
                    out_c = clients_kappa[k](h)
                    out_s = server_theta(h)
                    loss = GAMMA * F.cross_entropy(out_c, yb) + (1 - GAMMA) * F.cross_entropy(out_s, yb)
                    opt.zero_grad();
                    loss.backward();
                    opt.step()
            client_phi_states.append(deepcopy(clients_phi[k].state_dict()))
            client_kappa_states.append(deepcopy(clients_kappa[k].state_dict()))
            client_theta_states.append(deepcopy(server_theta.state_dict()))

        # aggregate server theta (FedAvg)
        new_theta = {}
        for key in client_theta_states[0].keys():
            new_theta[key] = sum(client_theta_states[i][key] for i in range(K)) / K
        theta_state = new_theta

        # client-side aggregation with lambda personalization
        # avg phi and kappa
        avg_phi = {}
        avg_kappa = {}
        for key in client_phi_states[0].keys():
            avg_phi[key] = sum(client_phi_states[i][key] for i in range(K)) / K
        for key in client_kappa_states[0].keys():
            avg_kappa[key] = sum(client_kappa_states[i][key] for i in range(K)) / K

        # new client-specific phi/kappa = lambda * local + (1-lambda) * avg
        new_client_phi_states = []
        new_client_kappa_states = []
        for i in range(K):
            new_phi = {k: lambda_personalization * client_phi_states[i][k] + (1 - lambda_personalization) * avg_phi[k]
                       for k in client_phi_states[i].keys()}
            new_kappa = {
                k: lambda_personalization * client_kappa_states[i][k] + (1 - lambda_personalization) * avg_kappa[k]
                for k in client_kappa_states[i].keys()}
            new_client_phi_states.append(new_phi)
            new_client_kappa_states.append(new_kappa)

        # set states for next round: pick phi_state and kappa_state to broadcast for initialization
        # we choose to broadcast avg (so phi_state is avg_phi and kappa_state avg_kappa) -- but clients will be set individually when training starts next round
        phi_state = avg_phi
        kappa_state = avg_kappa

        # after aggregation, load back into clients for next round (store per-client phi,kappa states)
        for i in range(K):
            clients_phi[i].load_state_dict(new_client_phi_states[i])
            clients_kappa[i].load_state_dict(new_client_kappa_states[i])
        server_theta.load_state_dict(theta_state)

    # final returned models: clients_phi (personalized per client), clients_kappa, server_theta
    return clients_phi, clients_kappa, server_theta


# -----------------------------
# Main experiment loop: run for p values and methods
# -----------------------------
def run_all():
    # Precompute per-client test loaders for each p value
    per_p_testsets = {}
    for p in p_values:
        print("p value:", p)
        per_client_testsets = [client_test_set_for_p(k, p) for k in range(K)]
        per_p_testsets[p] = per_client_testsets

    # print(f"Precomputed per-client test sets for all p values (per_p_testsets). : {per_p_testsets}")
    # Storage for results
    results = defaultdict(list)

    # 1) Personalized FL (local only)
    print("Training Personalized (local-only) FL...")
    personal_models = train_personalized_local_only()
    for p in tqdm(p_values,desc="Evaluating p values (Training Personalized (local-only) FL)"):
        avg_acc = evaluate_method_full_models(personal_models, per_p_testsets[p])
        results['Personalized'].append(avg_acc)
        print(f"p={p:.2f} Personalized avg acc: {avg_acc:.4f}")

    # # 2) Generalized FL (FedAvg)
    print("Training FedAvg global model...")
    global_model = train_fedavg_global()
    for p in tqdm(p_values,desc="Evaluating p values(Training FedAvg global model..)"):
        avg_acc = evaluate_method_full_models(global_model, per_p_testsets[p])
        results['Generalized-FedAvg'].append(avg_acc)
        print(f"p={p:.2f} FedAvg avg acc: {avg_acc:.4f}")

    # # 3) Multi-Exit NN via split with lambda=0 (no personalization)
    print("\n\nTraining Multi-Exit (split, lambda=0)...")
    clients_phi_me, clients_kappa_me, server_theta_me = train_split_training(lambda_personalization=0.0)
    for p in tqdm(p_values,desc="Evaluating p values (Training Multi-Exit (split, lambda=0)...)"):
        full_acc, client_acc = evaluate_method_split(clients_phi_me, clients_kappa_me, server_theta_me, per_p_testsets[p])
        # For fair global comparison, use full_acc (server-side predictions after offload)
        results['Multi-Exit (lambda=0)'].append(full_acc)
        print(f"p={p:.2f} Multi-Exit full avg acc: {full_acc:.4f} (client avg acc {client_acc:.4f})")

    # 4) SplitGP (lambda = 0.2)

    print("\n\nTraining SplitGP (lambda=0.2)...")
    clients_phi_sgp, clients_kappa_sgp, server_theta_sgp = train_split_training(lambda_personalization=LAMBDA_SPLITGP)
    for p in tqdm(p_values,desc="Evaluating p values (Training SplitGP (lambda=0.2)...)"):
        full_acc, client_acc = evaluate_method_split(clients_phi_sgp, clients_kappa_sgp, server_theta_sgp, per_p_testsets[p])
        results['SplitGP (lambda=0.2)'].append(full_acc)
        print(f"p={p:.2f} SplitGP full avg acc: {full_acc:.4f} (client avg acc {client_acc:.4f})")

    return results


# -----------------------------
# Run and plot
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # Config (change to scale)
    # -----------------------------



    # -----------------------------
    # Data: MNIST loaders
    # -----------------------------
    # transform = transforms.Compose([transforms.ToTensor()])
    #
    # trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    # testset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # -----------------------------
    # Datasets: CIFAR-10 loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    # initialize base network and split
    if isinstance(trainset, datasets.CIFAR10):
        IN_CHANNELS = 3
        IMG_SIZE = 32
    else:  # MNIST / FMNIST
        IN_CHANNELS = 1
        IMG_SIZE = 28

    print(f"Making Non IID datasets K={K}, shards={SHARDS}")
    clients_indices = create_clients_shards(trainset, K=K, shards=SHARDS)
    print(f"Created {K} clients; each client has approx {len(clients_indices[0])} train samples")

    # helper to read labels (works for torchvision dataset types)
    train_labels = np.array(trainset.targets)
    test_labels = np.array(testset.targets)
    results = run_all()
    plot_dir_name = 'plots'
    os.makedirs(plot_dir_name, exist_ok=True)

    results_df = pd.DataFrame(results, index=p_values)
    results_df.index.name = "p"
    csv_path = f"{plot_dir_name}/accuracy_vs_p.csv"
    results_df.to_csv(csv_path)
    print(f"Results saved to {csv_path}")


    # Plot
    plt.figure(figsize=(8, 6))
    for method, accs in results.items():
        plt.plot(p_values, accs, marker='o', label=method)
    plt.xlabel("p (relative portion of out-of-distribution test samples)")
    plt.ylabel("Average client test accuracy")
    plt.title("Accuracy vs p: Personalized FL / Generalized FL / Multi-Exit / SplitGP")
    plt.xticks(p_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # save before showing
    plt.savefig(f"{plot_dir_name}/accuracy_vs_p.png", dpi=300)  # PNG high-res
    plt.savefig(f"{plot_dir_name}/accuracy_vs_p.pdf")  # PDF vector graphics
    plt.show()
