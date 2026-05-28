# Math-for-AI-Final-Project-Alex-Gianni-
This is a repository to present the findings of the final project for the Math for AI Winter 2026 course at Vanier College in a polished, publicly accessible format.

# *1. Reproduce Grokking*

```import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

p = 97
fraction_train = 0.5

# Dataset
X = []
y = []
for a in range(p):
    for b in range(p):
        X.append([a, b])
        y.append((a + b) % p)

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Shuffle and split
indices = torch.randperm(len(X))
split_idx = int(fraction_train * len(X))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Model
class GrokkingMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, x):
        # x is (batch, 2)
        emb = self.embedding(x) # (batch, 2, embedding_dim)
        emb = emb.view(emb.size(0), -1) # (batch, 2 * embedding_dim)
        return self.mlp(emb)

model = GrokkingMLP(vocab_size=p, embedding_dim=128, hidden_dim=256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
criterion = nn.CrossEntropyLoss()

epochs = 15000
train_losses = []
test_losses = []
train_accs = []
test_accs = []

print("Starting training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            
            train_acc = (outputs.argmax(dim=1) == y_train).float().mean().item()
            test_acc = (test_outputs.argmax(dim=1) == y_test).float().mean().item()
            
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            print(f"Epoch {epoch:5d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}", flush=True)
            
            # Stop if test acc is perfect
            if test_acc >= 0.99 and train_acc >= 0.99 and epoch > 1000:
                print("Grokking complete! Perfect accuracy reached.")
                break

epochs_recorded = np.arange(len(train_losses)) * 100

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_recorded, train_losses, label='Train Loss', color='blue')
plt.plot(epochs_recorded, test_losses, label='Test Loss', color='red')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (Log Scale)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(epochs_recorded, train_accs, label='Train Acc', color='blue')
plt.plot(epochs_recorded, test_accs, label='Test Acc', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig('grokking_curves.png', dpi=300)
print("Plot saved to grokking_curves.png")
```

Antigravity created run_grokking.py, which trains a small embedding-based MLP on the modular addition task (a + b) mod 97. The model uses AdamW with strong weight decay, which helps produce the grokking effect. During training, it first memorized the training data: training accuracy quickly reached 100%, while validation accuracy stayed near 0%. Around epoch 1500, validation accuracy began improving sharply, and by about epoch 2300 it exceeded 99%. Finally, the script saved grokking_curves.png, showing the delayed jump in validation performance: <img width="1189" height="490" alt="download" src="https://github.com/user-attachments/assets/85504d85-da3f-4431-af93-e422ea49e549" />

# *2. Fourier Analysis*

```import torch
import matplotlib.pyplot as plt

model.eval()

device = next(model.parameters()).device
p = model.embedding.num_embeddings

print("Device:", device)
print("Prime p:", p)
print("Embedding layer:", model.embedding)
print("MLP:", model.mlp)

E = model.embedding.weight.detach().cpu()   # shape [p, embedding_dim]

print("Embedding matrix shape:", E.shape)

@torch.no_grad()
def get_hidden_activations_fixed_b(model, b_value=0):
    """
    Vary a from 0 to p-1 while fixing b.
    This creates a 1D signal over a.

    Input is [a, b].
    Target would be (a + b) mod p.
    """
    model.eval()

    a = torch.arange(p, device=device)
    b = torch.full_like(a, b_value)

    X_line = torch.stack([a, b], dim=1)

    emb = model.embedding(X_line)
    emb = emb.view(emb.size(0), -1)

    hidden_pre = model.mlp[0](emb)
    hidden = model.mlp[1](hidden_pre)

    return hidden.detach().cpu()


A = get_hidden_activations_fixed_b(model, b_value=0)

print("MLP hidden activation matrix shape:", A.shape)

def fourier_power(X):
    """
    X shape: [p, features]

    Applies 1D Fourier transform over the first axis,
    which corresponds to token/input value 0, 1, ..., p-1.
    """
    X = X.detach().cpu().float()

    X = X - X.mean(dim=0, keepdim=True)

    F = torch.fft.rfft(X, dim=0)
    power = F.real.pow(2) + F.imag.pow(2)

    return power


def summarize_spectrum(power, name, top_k=10):
    """
    power shape: [num_frequencies, features]
    """
    freq_power = power.sum(dim=1)
    total_power = freq_power.sum() + 1e-12

    top_vals, top_freqs = torch.topk(
        freq_power,
        k=min(top_k, len(freq_power))
    )

    probs = freq_power / total_power
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    normalized_entropy = entropy / torch.log(
        torch.tensor(len(probs), dtype=torch.float32)
    )

    print(f"\n=== {name} ===")
    print("Top frequencies:", top_freqs.tolist())
    print("Top-10 power fraction:", float(top_vals.sum() / total_power))
    print("Normalized spectral entropy:", float(normalized_entropy))

    return freq_power, top_freqs, top_vals

embedding_power = fourier_power(E)
mlp_power = fourier_power(A)

embedding_freq_power, embedding_top_freqs, embedding_top_vals = summarize_spectrum(
    embedding_power,
    "Embedding Matrix Fourier Spectrum",
    top_k=10
)

mlp_freq_power, mlp_top_freqs, mlp_top_vals = summarize_spectrum(
    mlp_power,
    "MLP Hidden Activation Fourier Spectrum",
    top_k=10
)

plt.figure(figsize=(8, 4))
plt.plot(embedding_freq_power.numpy())
plt.xlabel("Fourier frequency k")
plt.ylabel("Total spectral power")
plt.title("Embedding Matrix Fourier Power")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(mlp_freq_power.numpy())
plt.xlabel("Fourier frequency k")
plt.ylabel("Total spectral power")
plt.title("MLP Hidden Activation Fourier Power")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

normalized_mlp_power = mlp_power / (mlp_power.sum(dim=0, keepdim=True) + 1e-12)

vals, freqs = torch.topk(
    normalized_mlp_power,
    k=3,
    dim=0
)

print("\n=== Individual MLP Neuron Frequency Selectivity ===")

for neuron in range(min(30, normalized_mlp_power.shape[1])):
    print(
        f"Neuron {neuron:03d} | "
        f"top frequencies: {freqs[:, neuron].tolist()} | "
        f"power fractions: {[round(v, 4) for v in vals[:, neuron].tolist()]}"
    )


plt.figure(figsize=(10, 5))
plt.imshow(
    normalized_mlp_power[:, :128].numpy(),
    aspect="auto",
    origin="lower"
)
plt.xlabel("Neuron index")
plt.ylabel("Fourier frequency k")
plt.title("MLP Neuron Fourier Selectivity")
plt.colorbar(label="Fraction of neuron power")
plt.tight_layout()
plt.show()
```
The learned embedding matrix and MLP activations have Fourier spectra whose power is concentrated on a small number of frequencies. The top-k Fourier modes explain a large fraction of total spectral power, and the normalized spectral entropy is low compared with a randomly initialized control. This indicates that the network has learned to represent the input domain using a sparse set of key frequencies. <img width="789" height="390" alt="image" src="https://github.com/user-attachments/assets/f9f366cf-ada6-44b1-80ca-f8676b5ea7f5" /> <img width="789" height="390" alt="image" src="https://github.com/user-attachments/assets/4934d0a7-8a32-489b-96a0-558c9aa83783" /> <img width="923" height="490" alt="image" src="https://github.com/user-attachments/assets/6ce616f7-d04d-471b-8a3b-fe6be5da1084" />

# *3. Memorization, Circuit Formation & Analysis*

```
```

# *4. Epochs Until Generalization*

```import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = 97
fractions = [0.10, 0.30, 0.60, 0.90]
max_epochs = 15000
check_every = 100
threshold = 0.99

X = []
y = []

for a in range(p):
    for b in range(p):
        X.append([a, b])
        y.append((a + b) % p)

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

class GrokkingMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(emb.size(0), -1)
        return self.mlp(emb)

def train_one_fraction(fraction_train):
    torch.manual_seed(42)
    np.random.seed(42)

    indices = torch.randperm(len(X))
    split_idx = int(fraction_train * len(X))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx].to(device)
    y_train = y[train_idx].to(device)
    X_test = X[test_idx].to(device)
    y_test = y[test_idx].to(device)

    model = GrokkingMLP(
        vocab_size=p,
        embedding_dim=128,
        hidden_dim=256
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1.0
    )

    criterion = nn.CrossEntropyLoss()

    train_accs = []
    test_accs = []
    epochs = []

    memorization_epoch = None
    generalization_epoch = None

    print()
    print("Training fraction:", fraction_train)

    for epoch in range(max_epochs + 1):
        model.train()

        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % check_every == 0:
            model.eval()

            with torch.no_grad():
                train_logits = model(X_train)
                test_logits = model(X_test)

                train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()
                test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epochs.append(epoch)

            print(
                "Epoch",
                epoch,
                "| Train Acc:",
                round(train_acc, 4),
                "| Test Acc:",
                round(test_acc, 4)
            )

            if memorization_epoch is None and train_acc >= threshold:
                memorization_epoch = epoch

            if generalization_epoch is None and test_acc >= threshold:
                generalization_epoch = epoch
                break

    if memorization_epoch is None:
        grokking_delay = None
    elif generalization_epoch is None:
        grokking_delay = None
    else:
        grokking_delay = generalization_epoch - memorization_epoch

    return {
        "fraction": fraction_train,
        "epochs": epochs,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "memorization_epoch": memorization_epoch,
        "generalization_epoch": generalization_epoch,
        "grokking_delay": grokking_delay
    }

results = []

for fraction in fractions:
    results.append(train_one_fraction(fraction))

print()
print("Summary")

for r in results:
    print(
        "Fraction:",
        r["fraction"],
        "| Memorization epoch:",
        r["memorization_epoch"],
        "| Generalization epoch:",
        r["generalization_epoch"],
        "| Grokking delay:",
        r["grokking_delay"]
    )

x = [100 * r["fraction"] for r in results]

y_generalization = [
    r["generalization_epoch"] if r["generalization_epoch"] is not None else max_epochs
    for r in results
]

labels_generalization = [
    str(r["generalization_epoch"]) if r["generalization_epoch"] is not None else "not reached"
    for r in results
]

plt.figure(figsize=(8, 5))
plt.plot(x, y_generalization, marker="o")
plt.xlabel("Training Data Fraction")
plt.ylabel("Epochs until Generalization")
plt.title("Training Fraction vs Epochs until Generalization")
plt.grid(True, alpha=0.3)

for xi, yi, label in zip(x, y_generalization, labels_generalization):
    plt.text(xi, yi, label, ha="center", va="bottom")

plt.tight_layout()
plt.savefig("epochs_until_generalization.png", dpi=300)
plt.show()

plt.figure(figsize=(9, 5))

for r in results:
    plt.plot(
        r["epochs"],
        r["test_accs"],
        label=str(int(100 * r["fraction"])) + " percent train"
    )

plt.axhline(threshold, linestyle="--", label="generalization threshold")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Across Training Data Fractions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_accuracy_by_fraction.png", dpi=300)
plt.show()

delay_values = [
    r["grokking_delay"] if r["grokking_delay"] is not None else 0
    for r in results
]

delay_labels = [
    str(r["grokking_delay"]) if r["grokking_delay"] is not None else "no grokking"
    for r in results
]

plt.figure(figsize=(8, 5))
plt.bar(x, delay_values, width=10)
plt.xlabel("Training Data Fraction")
plt.ylabel("Generalization Epoch minus Memorization Epoch")
plt.title("Grokking Delay by Training Data Fraction")
plt.grid(True, axis="y", alpha=0.3)

for xi, yi, label in zip(x, delay_values, delay_labels):
    plt.text(xi, yi, label, ha="center", va="bottom")

plt.tight_layout()
plt.savefig("grokking_delay_by_fraction.png", dpi=300)
plt.show()

print()
print("Saved epochs_until_generalization.png")
print("Saved test_accuracy_by_fraction.png")
print("Saved grokking_delay_by_fraction.png")
```

Antigravity's code varied the amount of training data used for the modular addition task ((a + b) \bmod 97), using 10%, 30%, 60%, and 90% of the full dataset. For each fraction, it trained the same embedding-based MLP with AdamW and strong weight decay, then recorded when the model reached 99% validation accuracy. The first graph shows that with only 10% training data, the model never generalized within 15,000 epochs. With 30% training data, it eventually generalized, but only after a very long delay around epoch 10,900. With 60% and 90% training data, generalization happened much faster, around epochs 1,600 and 700. The second graph shows the validation accuracy curves directly: 30% has the clearest grokking pattern because validation accuracy stays near zero for thousands of epochs before suddenly rising, while 60% and 90% improve much earlier. The third graph measures the grokking delay, meaning the time between memorizing the training set and generalizing to the test set. The delay is largest at 30%, smaller at 60% and 90%, and absent at 10% because the model never generalized. In all, the plots show that grokking appears most strongly in a specific data-scarce regime, too little data prevents generalization, moderate scarcity creates delayed grokking, and abundant data makes generalization happen quickly: <img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/fccd1d55-13e7-455a-83ee-985ed56ed15f" /> <img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/22e75c48-74da-4cca-bdb8-3234ec8ea1e3" /> <img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/f8b2d2cc-5b01-4d10-b60d-c2db078e42b5" />




# *6. Algorithm Teasing Small Transformer*

```
```
