import torch
import optuna
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib.pyplot as plt

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

dataset = load_dataset("squad", split="train[:1000]")
def tokenize_data(example):
    prompt = example["question"].strip()
    label = example["answers"]["text"][0].strip() if example["answers"]["text"] else ""
    full_text = prompt + " " + label
    return {"full_text": full_text, "label_length": len(tokenizer(label)["input_ids"])}
tokenized_data = [tokenize_data(q) for q in dataset]

# Precompute gradient vectors
def compute_sample_gradient(full_text, label_length):
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    labels[:, :-label_length] = -100
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    grad_vector = torch.cat(grads)
    model.zero_grad()
    return grad_vector.cpu().detach()

print("Precomputing gradients...")
precomputed_gradients = [compute_sample_gradient(ex["full_text"], ex["label_length"]) for ex in tokenized_data]

# Eigenvalue-based sinusoidal scoring
def sinusoidal_weighted_score(batch_indices):
    batch_grads = torch.stack([precomputed_gradients[i] for i in batch_indices])
    H = torch.matmul(batch_grads, batch_grads.T) / len(batch_indices)
    eigenvalues = torch.linalg.eigvalsh(H)  # Sorted eigenvalues (real)
    
    n = len(eigenvalues)
    weights = torch.sin(torch.linspace(0, np.pi, n))  # sin weights: peak at ends, zero in middle
    final_score = torch.sum(weights * eigenvalues)
    return final_score.item()

# Optuna objective using sinusoidal eigenvalue score
def objective(trial):
    batch_indices = [trial.suggest_int(f"sample_{i}", 0, len(tokenized_data)-1) for i in range(batch_size)]
    return sinusoidal_weighted_score(batch_indices)

# Training loop
def adaptive_train(epochs=30, reoptimize_every=5):
    loss_history = []
    best_batches = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1} started...")
        model.train()
        total_loss = 0

        if epoch % reoptimize_every == 0:
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=20)
            best_batch = [study.best_params[f"sample_{i}"] for i in range(batch_size)]
            best_batches.append(best_batch)
        
        selected_data = [tokenized_data[i] for i in best_batches[-1]]

        for ex in selected_data:
            inputs = tokenizer(ex["full_text"], return_tensors="pt").to(model.device)
            labels = inputs["input_ids"].clone()
            labels[:, :-ex["label_length"]] = -100
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(selected_data)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return loss_history, best_batches

# Training comparison
def standard_train(epochs=30):
    loss_history = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1} started...")
        model.train()
        total_loss = 0
        indices = np.random.choice(len(tokenized_data), size=batch_size, replace=False)
        selected_data = [tokenized_data[i] for i in indices]
        for ex in selected_data:
            inputs = tokenizer(ex["full_text"], return_tensors="pt").to(model.device)
            labels = inputs["input_ids"].clone()
            labels[:, :-ex["label_length"]] = -100
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(selected_data)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return loss_history

batch_size = 5
adaptive_loss_history, _ = adaptive_train()
standard_loss_history = standard_train()

# Plotting results
plt.plot(adaptive_loss_history, label="Eigen-Sinusoidal BO", marker='o')
plt.plot(standard_loss_history, label="Random Sampling", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Sinusoidal Eigenvalue Selection vs Random")
plt.legend()
plt.show()
