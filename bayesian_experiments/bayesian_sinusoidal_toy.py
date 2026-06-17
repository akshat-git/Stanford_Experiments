import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
import random
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler

# === Reproducibility ===
SEED = 42
# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# optuna.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === Toy Dataset ===
toy_texts = [
    "the cat sat",
    "the dog ran",
    "the bird flew",
    "the fish swam",
    "a cat ran",
    "a dog barked",
    "a bird sang",
    "the cat slept"
]

vocab = sorted(set(" ".join(toy_texts).split()))
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for w, i in stoi.items()}
vocab_size = len(vocab)

def encode(text): return [stoi[w] for w in text.split()]
def decode(ids): return " ".join([itos[i] for i in ids])
dataset = [encode(line) for line in toy_texts]

# === Tiny Transformer Model ===
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, nlayers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64),
            num_layers=nlayers
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.head(x)

# === Compute gradient of loss w.r.t. parameters ===
def compute_gradient(seq, model):
    model.zero_grad()
    x = torch.tensor(seq[:-1]).unsqueeze(1)
    y = torch.tensor(seq[1:]).unsqueeze(1)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return grad.detach()

# === Sinusoidal Score Function ===
def sinusoidally_scaled_score(batch_indices, model):
    G = torch.stack([compute_gradient(dataset[i], model) for i in batch_indices])
    n = G.size(0)
    weights = torch.sin(torch.linspace(0, torch.pi, n))
    G_scaled = weights.unsqueeze(1) * G
    C = G_scaled.T @ G_scaled / n
    eigvals, eigvecs = torch.linalg.eigh(C)
    return eigvals[-1].item(), eigvecs[:, -1]

# === Optuna Objective with Gradient Caching ===
def make_objective(model, batch_size):
    grad_cache = {}

    def objective(trial):
        indices = [trial.suggest_int(f"sample_{i}", 0, len(dataset)-1) for i in range(batch_size)]
        G = []
        for i in indices:
            if i not in grad_cache:
                grad_cache[i] = compute_gradient(dataset[i], model)
            G.append(grad_cache[i])
        G = torch.stack(G)
        n = G.size(0)
        weights = torch.sin(torch.linspace(0, torch.pi, n))
        G_scaled = weights.unsqueeze(1) * G
        C = G_scaled.T @ G_scaled / n
        eigvals, _ = torch.linalg.eigh(C)
        return -eigvals[-1].item()

    return objective

# === Training Loop ===
def train(model, use_optuna=True, label="PCA", epochs=40, reoptimize_every=2, step_size=0.001, batch_size=2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_history = []
    best_batches = []

    for epoch in range(epochs):
        print(f"[{label}] Epoch {epoch+1}")

        if use_optuna and epoch % reoptimize_every == 0:
            study = optuna.create_study(
                direction="minimize",
                sampler=TPESampler(n_startup_trials=1, seed=SEED)
            )
            study.optimize(make_objective(model, batch_size), n_trials=10)
            best_batch = [study.best_params[f"sample_{i}"] for i in range(batch_size)]
            best_batches.append(best_batch)

            # Recompute top eigenvector from best batch
            _, top_vec = sinusoidally_scaled_score(best_batch, model)
            top_vec = top_vec / top_vec.norm()
            i = 0
            with torch.no_grad():
                for p in model.parameters():
                    sz = p.numel()
                    p.add_(top_vec[i:i+sz].view_as(p) * step_size)
                    i += sz

        elif not use_optuna:
            best_batch = random.sample(range(len(dataset)), batch_size)
            best_batches.append(best_batch)

        # === Build final batch of exactly 5 unique prompts ===
        merged_batch = sum(best_batches[-5:], [])  # flatten recent best batches
        merged_batch = list(dict.fromkeys(merged_batch))  # remove duplicates

        while len(merged_batch) < 5:
            candidate = random.choice(range(len(dataset)))
            if candidate not in merged_batch:
                merged_batch.append(candidate)

        merged_batch = merged_batch[:5]  # cap to 5

        # === Train on merged batch ===
        total_loss = 0
        for idx in merged_batch:
            seq = dataset[idx]
            x = torch.tensor(seq[:-1]).unsqueeze(1)
            y = torch.tensor(seq[1:]).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(merged_batch)
        loss_history.append(avg_loss)
        print(f"  ↪ Loss: {avg_loss:.4f}")

    return loss_history

# === Run both PCA and Random training ===
batch_size = 2
pca_model = TinyTransformerLM(vocab_size).cpu()
rand_model = TinyTransformerLM(vocab_size).cpu()

pca_losses = train(pca_model, use_optuna=True, label="PCA", batch_size=batch_size)
rand_losses = train(rand_model, use_optuna=False, label="Random", batch_size=batch_size)

# === Plot Comparison ===
plt.plot(pca_losses, label="PCA Sinusoidal Selection", marker='o')
plt.plot(rand_losses, label="Random Selection", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Sinusoidal Gradient PCA vs Random Batch Training (Fixed 5-Prompt)")
plt.legend()
plt.grid(True)
plt.show()
