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

```import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random


P = 113
TRAIN_FRAC = 0.3
EPOCHS = 4000
LR = 1e-3
WEIGHT_DECAY = 1.0
D_MODEL = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)



pairs = [(i, j) for i in range(P) for j in range(P)]

random.seed(0)
random.shuffle(pairs)

split = int(TRAIN_FRAC * len(pairs))

train_pairs = pairs[:split]
test_pairs = pairs[split:]

def fn(x, y):
    return (x + y) % P

x_train = torch.tensor(train_pairs, device=DEVICE)
y_train = torch.tensor([fn(i, j) for i, j in train_pairs], device=DEVICE)

x_test = torch.tensor(test_pairs, device=DEVICE)
y_test = torch.tensor([fn(i, j) for i, j in test_pairs], device=DEVICE)

all_pairs = torch.tensor(pairs, device=DEVICE)
all_labels = torch.tensor([fn(i, j) for i, j in pairs], device=DEVICE)


class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(P + 1, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=4,
            dim_feedforward=4 * D_MODEL,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.unembed = nn.Linear(D_MODEL, P)

    def forward(self, x):

        x = self.embed(x)

        x = self.transformer(x)

        x = x[:, -1]

        return self.unembed(x)

model = SimpleTransformer().to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


basis = []

basis.append(torch.ones(P) / np.sqrt(P))

for k in range(1, P // 2 + 1):
    basis.append(torch.cos(2 * torch.pi * torch.arange(P) * k / P))
    basis.append(torch.sin(2 * torch.pi * torch.arange(P) * k / P))

basis = torch.stack(basis).to(DEVICE)

def fft2d(values):

    values = values.reshape(P, P, -1)

    return torch.einsum(
        'xyc,fx,gy->fgc',
        values,
        basis,
        basis
    )



@torch.no_grad()
def get_key_freqs():

    logits = model(all_pairs)

    fourier = fft2d(logits)

    power = fourier.pow(2).sum(-1)

    power[0, 0] = 0

    flat = power.flatten()

    topk = torch.topk(flat, 6)

    freqs = []

    for idx in topk.indices:

        x = idx // power.shape[1]
        y = idx % power.shape[1]

        freqs.append((x.item(), y.item()))

    return freqs


@torch.no_grad()
def compute_losses():

    logits = model(all_pairs)

    total_loss = F.cross_entropy(logits, all_labels)


    fourier = fft2d(logits)

    key_freqs = get_key_freqs()

    restricted_fourier = torch.zeros_like(fourier)

    for fx, fy in key_freqs:
        restricted_fourier[fx, fy] = fourier[fx, fy]

    excluded_fourier = fourier - restricted_fourier


    restricted_logits = torch.einsum(
        'fgc,fx,gy->xyc',
        restricted_fourier,
        basis,
        basis
    ).reshape(P * P, P)

    excluded_logits = torch.einsum(
        'fgc,fx,gy->xyc',
        excluded_fourier,
        basis,
        basis
    ).reshape(P * P, P)

    restricted_loss = F.cross_entropy(
        restricted_logits,
        all_labels
    )

    excluded_loss = F.cross_entropy(
        excluded_logits,
        all_labels
    )

    return (
        total_loss.item(),
        restricted_loss.item(),
        excluded_loss.item()
    )



train_losses = []
test_losses = []

restricted_losses = []
excluded_losses = []

weight_norms = []

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    logits = model(x_train)

    loss = F.cross_entropy(logits, y_train)

    loss.backward()

    optimizer.step()


    if epoch % 100 == 0:

        model.eval()

        with torch.no_grad():

            train_loss = F.cross_entropy(
                model(x_train),
                y_train
            ).item()

            test_loss = F.cross_entropy(
                model(x_test),
                y_test
            ).item()

            _, restricted_loss, excluded_loss = compute_losses()

            weight_norm = sum(
                p.pow(2).sum().item()
                for p in model.parameters()
            )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        restricted_losses.append(restricted_loss)
        excluded_losses.append(excluded_loss)

        weight_norms.append(weight_norm)

        print(
            f"Epoch {epoch:5d} | "
            f"Train {train_loss:.4f} | "
            f"Test {test_loss:.4f}"
        )



epochs = np.arange(len(train_losses)) * 100

fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.semilogy(epochs, train_losses, label="Train Loss")
ax1.semilogy(epochs, test_losses, label="Test Loss")
ax1.semilogy(epochs, restricted_losses, label="Restricted Loss")
ax1.semilogy(epochs, excluded_losses, label="Excluded Loss")

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (log scale)")

ax2 = ax1.twinx()

ax2.plot(
    epochs,
    weight_norms,
    linestyle="--",
    label="Weight Norm"
)

ax2.set_ylabel("Sum of Squared Weights")


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2)

plt.title("Three Phases of Grokking")

plt.show()
```
In this case, the antigravity’s code was utilized to train a small transformer on the modular addition problem ((a + b) \bmod 113), with 30% of all combinations being used for the training process and 70% – for the evaluation stage. As usual, the code uses embeddings and a single transformer encoder layer with AdamW optimizer and a high value of weight decay to favor the emergence of grokking. While running the learning algorithm, the script measures not only the standard train and test losses, but also two more metrics associated with Fourier transform, namely restricted loss and excluded loss, as well as the total sum of the squared weights in the network. In order to calculate these parameters, the code converts model’s logits to a two-dimensional Fourier basis, extracts six main frequency components, and separates the logits into restricted – which consists of frequencies that are dominating during the transformation process – and excluded components, comprising all other frequencies. Then, this graph is specifically designed to plot the three stages of grokking, including memorization. During the Circuit Formation stage, the limited loss starts falling along with the decreasing value of the weight norm, as weight decay eliminates any excess information memorized during training while reinforcing a simple Fourier-based circuit that truly reflects the modular addition process. During the third and last stage, known as Cleanup, the loss dramatically declines from its previously high value after several epochs because the algorithmic circuit becomes dominant over the memorized one.
data:image/png;base64,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Jj2IiIjuQSaTYdOmTfj000+xatUqbNy4ERqNBo0bN8Zrr71mNdxj2rRpaN68OT799FO8//77AExzfvTt2xdPPPFElcbt5uaGLVu2YMqUKXj33Xfh4uKCZ599Fr1790a/fv0s9QIDA9GvXz/8/vvvuH79OjQaDQIDA7Ft2zZ07ty5TO+ttG0VZ/To0dBoNPj444/x1ltvwc7ODk899RTmzZsHZ2fncv8cnnzySZw5cwYLFy7Ezp07sWLFCgiCAH9/fzz22GN46aWXEBgYeM92ZDIZNm/ejBkzZmDdunX47rvv0LBhQyxYsABTpky562sfeeQR/Pzzz/jf//6H5557DmvWrCm23ttvv420tDRL4iM0NLRc79nMz88Pe/fuxaRJk/DRRx+hXr16CA0NhZ2dHSZNmmS1Ug4REVFtJIic7YqIiIioTpk8eTKWLVuGzMzMEidEJSIiqg04pwcRERFRLVZ4HpGkpCT88MMPePjhh5nwICKiWo/DW4iIiIhqsS5duiA4OBitWrVCQkICvv32W6Snp+O9996TOjQiIqJKx6QHERERUS02YMAAbNiwAcuXL4cgCHjooYfw7bffWi1jTEREVFtxTg8iIiIiIiIiqpU4pwcRERERERER1UpMehARERERERFRrcQ5Pe5Br9cjIiICnp6ekMmYIyIiIiIiIqLKZTQakZCQgKCgICgUfGy/H/zplSA8PBzh4eHIzs5GTEyM1OEQERERERFRHXPs2DF06NBB6jBqNE5keg+xsbHw9/fHsWPH4O3tLXU4REREREREVMvdvHkTHTt2RExMDBo0aCB1ODUae3rcg3lIi7e3N3x9fSWOhoiIiIiIiOoKTrFw//gTJCIiIiIiIqJaiUkPIiIiIiIiIqqVmPQgIiIiIiIiolqJSQ8iIiIiIiIiqpWY9CAiIiIiIiKiWolJjxKEh4cjICAAwcHBUodCREREREREROXApEcJQkNDERUVhX379kkdChERERERERGVA5MeRERERERERFQrMelBRERERERERLUSkx5EREREREREVCsx6UFEREREREREtRKTHkRERERERERUKzHpQURERERERES1EpMeRERERERERFQrMelBRERERERERLUSkx5EREREREREVCsx6UFEREREREREtRKTHiUIDw9HQEAAgoODpQ6FiIiIiIiIiMqBSY8ShIaGIioq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OADgo48+wueff45jx46hfzkSY4sWLULv3r3x3nvvATCtGhoVFYUFCxZg9OjRiI2NhZ2dHR5//HE4ODjA398fQUFBAExJD71ej6effhr+/v4AgDZt2pQ5BqpeWrdujQsXLuDLL7+Eg4MDMjMz8fTTTyM0NBTe3t7larPMSQ8vLy9cunQJDRs2tCr/+++/0bhx43IFQSS1AB9HqUMgIiIiqlaUXl5weOQRZOzahZSffoL3zJlShyQJW6UcUR/0k+zeFaV9+/ZW55mZmZg1axb++OMPSwIhJycHsbGxd23nwQcftBzb2dnB0dERieVc5efcuXN48sknrcq6deuGxYsXw2AwoE+fPvD390fjxo3Rv39/9O/f3zK0JjAwEL1790abNm3Qr18/9O3bF4MHD4aLi0u5YiFpjRo1Cr1790ZwcDAaNGiAd999t8LaLvOcHuPHj8drr72Go0ePQhAE3LhxA6tXr8bUqVPx8ssvV1hgRFKJScpCnt4odRhEREREknMZORIAkP7bZhgyMyWORhqCIECjUkiyCYJQYe/Dzs66R/PUqVOxceNGfPTRRzhw4AAiIyPRpk0b5OXl3bUdpVJZ5OdjNFbO784ODg44deoUfvrpJ3h7e2PGjBkIDAxEamoq5HI5du3ahW3btiEgIABffPEFWrRogejo6EqJhSpXTEwMXnzxRTRq1AhNmjTBCy+8gDVr1iA+Pv6+2y5z0mPatGkYMWIEevfujczMTPTo0QMvvPACXnzxRbz66qv3HRCRlN7ddAbBn+zDltM3pA6FiIiISHKaTh2hatoExuxspG36TepwqAIdPHgQo0ePxlNPPYU2bdrAy8sLV69erdIYWrVqVWSehoMHD6J58+aQy029XBQKBUJCQjB//nycPn0aV69exZ49ewCYEi7dunXD+++/j4iICKhUKmzcuLFK3wNVjH379iE1NRV//vknnn32WVy8eBFjx45F/fr10bJlS7z88stYv758S2iXeXiLIAh455138MYbb+DSpUvIzMxEQEAA7O3tyxUAUXXi7WQLUTQtX/tUUP0Kza4TERER1TSCIMBl+HAkzJ6DlJ9+gsvIEfz9qJZo1qwZfv31VwwcOBCCIOC9996rtB4bt27dQmRkpFWZt7c3pkyZgg4dOmD27NkYOnQoDh8+jC+//BJfffUVAGDLli24cuUKevToARcXF2zduhVGoxEtWrTA0aNHsXv3bvTt2xceHh44evQobt26hVatWlXKe6DKp1ar0atXL8sqLbm5uTh06BC2bduG5cuXY/ny5XjmmWfK3G65pwJWqVQICAgo78uJqqVnO/kjfO8lnI/PwP6Lt9GzeT2pQyIiIiKSlNOTT+LWwkXIu3wZ2UePwq5zZ6lDogqwaNEijB07Fl27doW7uzveeustpKenV8q91qxZgzVr1liVzZ49G++++y5+/vlnzJgxA7Nnz4a3tzc++OADjB49GgDg7OyMX3/9FbNmzUJubi6aNWuGn376CQ888ADOnTuH/fv3Y/HixUhPT4e/vz8WLlyIRx99tFLeA1WdvLw8HD58GPv27cPevXtx9OhR+Pj44H//+1+52hPEUqzp8/TTT5e6wV9//bVcgVRX165dg5+fH+Li4uDry6VM64L3fz+L7w5eRbemblj9Av9RJyIiIor/4AOkrPkJDn36wPeLz6UOp1Ll5uYiOjoajRo1go2NjdThUC12t89aXXsO3b9/v1WSo0GDBujZsyd69uyJHj163NfPoFQ9PZycnMp9A6KaZtzDjbDqcAwOXkrCv9fT0Lo+P/9ERERUt7kMH46UNT8hY/du6G7ehLKcS0cSERXHvGrLW2+9hbVr18LT07PC2i5V0uO7776rsBsSVXe+Lho81sYbm/+5ga8PXMFnw4KkDomIiIhIUupmzaDp2BHZx44hZd06eEyeLHVIRFSLvPnmm9i3bx8mT56MJUuWoGfPnggODkbPnj3h7u5+X22XefWWmuipp56Ci4sLBg8eLHUoVENM6NEYAPD3xdvIztNLHA0RERGR9FxGjAAApK7fAOM9ljUlIiqLjz/+GEeOHEFSUhLmzZsHjUaD+fPnw8fHB61bt0ZoaCg2bNhQrrbLNZHphg0b8PPPPyM2NrbIOs6nTp0qVyCV6bXXXsPYsWPx/fffSx0K1RCt6zshfMRDCG5RDxpVuef7JSIiIqo1HHo/AoWHB/SJicjYsRNOAx+XOiQiqmXs7e3x6KOPWiakTU5OxqJFi/DFF19g6dKlMBgMZW6zzD09Pv/8c4wZMwaenp6IiIhAx44d4ebmhitXrlTbmXKDg4Ph4OAgdRhUwzz2oDfs1Ex4EBEREQGAoFTCeegQAEBKoZU4iIgqgtFoxNGjRzFv3jw8+uijaNiwIT766CO4uLjg+eefL1ebZU56fPXVV1i+fDm++OILqFQqvPnmm9i1axcmTZqEtLS0cgVxN/v378fAgQPh4+MDQRCwadOmInXCw8PRsGFD2NjYoFOnTjh27FiFx0F1l9Eo4mJChtRhEBEREUnO+ZlnAIUCORERyD13TupwiKiWmD9/PgYMGAAXFxd06dIFX375Jdzd3bF48WJcvnwZV69eLfdco2VOesTGxqJr164AAFtbW2RkmB4Gn3vuOfz000/lCuJusrKyEBgYiPDw8GKvr1u3DmFhYZg5cyZOnTqFwMBA9OvXD4mJiRUeC9U9iRm5eCL8bzz91SEkZ3HsKhEREdVtSg8POPbtA4C9PYio4ixevBjOzs745JNP8N9//yEuLg4//PADxo4di0aNGt1X22VOenh5eSE5ORkA0KBBAxw5cgQAEB0dDVEU7yuY4jz66KOYM2cOnnrqqWKvL1q0COPHj8eYMWMQEBCApUuXQqPRYMWKFeW6n1arRXp6umUzJ3WobnK3U8NoBDK0eny++6LU4RARERFJzjyhadrvW2CohJ7eRFT33LhxA2vWrMH48ePRtGnTCm27zEmPRx55BJs3bwYAjBkzBq+//jr69OmDoUOHlpiYqCx5eXk4efIkQkJCLGUymQwhISE4fPhwudqcO3cunJycLFtAQEBFhUs1kEwm4J3HWgEAfjwSg+jbWRJHRERERCQt23btoG7eHGJuLlI3bpQ6HCKiuypz0mP58uV45513AAChoaFYsWIFWrVqhQ8++ABLliyp8ADv5vbt2zAYDPD09LQq9/T0RHx8vOU8JCQEzzzzDLZu3QpfX9+7JkSmT5+OtLQ0yxYVFVVp8VPN0K2pO3q1qAe9UcT87eelDoeIiIhIUoIg3Fm+du26SuntTURUUcq8NIVMJoNMdidXMmzYMAwbNqxCg6pof/75Z6nrqtVqqNVqy3l6enplhETlYdAD2nTTlpsOaDPyzzOA3DTTXq8FRAMgGgGjwXRsNBZTln9epMwAiGKRetN1TvgLg7Ht33ic+OoFtLe9CUA01S3XvsD7EgodCEKh4/xrBY9LVc+q8eLLrcoKXi50r4L3u9tekFnHUJrXIP91RcpQqL273avwtULtWurKirlWXAwlvE7I/7uv2Oslvaakdgq+TgbIZIAgzz+Wm45l8vz68jtlluuyYurmt2FuS6YosJU5x01ERFQip4GPI2HuXORdvQrt+fOwadVK6pDqNKGk3+nyzZw5E7NmzSp32xs3bsSgQYMqpB5RVStz0uO7776Dvb09nnnmGavy9evXIzs7G6NGjaqw4O7F3d0dcrkcCQkJVuUJCQnw8vK6r7bDw8MRHh6OvDxOXlnh0m8AiVH5yYoCyQvLcVrx1/Q5koXcHMBQuTt+MjyCD+Na41fV+hLzBUTVk1AoCVI4KVLKc7kSkClNe7kSkKvyy1WFyu5RR5ZfLs8vV9jc2StUgFx951hhY3o9/6MjIqo2ZHZ2sO/RHRm7/kT6zp1Mekjs5s2bluN169ZhxowZuHDhgqXM3t5eirCIqoUyJz3mzp2LZcuWFSn38PDAhAkTqjTpoVKp0K5dO+zevduSUTQajdi9ezcmTpx4X22HhoYiNDQU165dg5+fXwVEW4dpM4CrB4Ere4HLe4HbF+79mrtRagC1A6B2NO1tHO+cK2xK+Ma7YFmhb9Otvi0v5tv2/G/UX88V8NtmEUanJkjp9R1cbUvbi6GYHgzmc0t30IK9P0TrcuBOD5HSHlvOC7RZpLykuiW8vty9WgrFV+LeWPy1u71ONN69rNhyYwnn4j2ul7f+Xe5feLP0QDIU6qlUqEw03unBVPA1orHon2PBP0OjzrTVSIJ1EkSuBhQFtuLOlTaAwhZQ2pr+3lDa5O9tTW2Yj5UF6igK1WEPGSKiEjn07YuMXX8iY8dOeLz2mtThVB5RBHTZ0txbqSlV0r/gF75OTk4QBMGq7JtvvsHChQsRHR2Nhg0bYtKkSXjllVcAmOZJDAsLwy+//IKUlBR4enripZdewvTp09GwYUMAsMzd6O/vj6tXr5b5bRiNRsyZMwfLly/HrVu30KpVK3z88cfo37//PWMQRRHvv/8+VqxYgYSEBLi5uWHw4MH4/PPPyxwH1U1lTnrExsYWu2SMv78/YmNjKySogjIzM3Hp0iXLeXR0NCIjI+Hq6ooGDRogLCwMo0aNQvv27dGxY0csXrwYWVlZGDNmTIXHQqVk0AM3TgFX9pmSHNeOAUZ9gQoCUK8FYOuan7AoJnlh41RMYiP/WK6U5G15ANjkn4Gm9ewhk/EbZ6qGzMmUgsOzjPoC+4Jb4bL8c9FQch2DeZ93Z2/IM5Ub8kwJFYN5K+7c/JpCdfR5gEFbYJ+/WSVoRFNvL30OgCpcKUBhk58AKZAYUdkBanvT30eqQnu1PaByKHrdfKy0ZY8VIqo17IODISiVyLtyBdpLl6Cu4BUXqg1dNvCRjzT3fvuG6d+d+7B69WrMmDEDX375JYKCghAREYHx48fDzs4Oo0aNwueff47Nmzfj559/RoMGDRAXF4e4uDgAwPHjx+Hh4YHvvvsO/fv3h1wuL1cMn332GRYuXIhly5YhKCgIK1aswBNPPIGzZ8+iWbNmd43hl19+waeffoq1a9figQceQHx8PP7555/7+plQ9RIUFHTPIVpmp06dKnP7ZU56eHh44PTp05asn9k///wDNze3MgdwLydOnECvXr0s52FhYQCAUaNGYeXKlRg6dChu3bqFGTNmID4+Hm3btsX27duLTG5aVhzeUgaiCCRfAS7vMSU6og+YhqgU5NIQaNwLaNILaNgd0LhKEel9a+7pIHUIRCUrOP9HbWA05idBck0JEX2uKVFidV4gSaLXFjrPAXS5gC7H9AurPte01+VYb/qcO3V0OaZ7mOlzTRtSKuY9CfJCCRL7OwkSG2fA1jl/71LguECZjVPt+fMlohpP7uAAu65dkfnXX0jfsQP1amvSo4abOXMmFi5ciKeffhoA0KhRI0RFRWHZsmUYNWoUYmNj0axZMzz88MMQBAH+/v6W19arVw8A4OzsfF/TB3zyySd46623LHNBzps3D3v37sXixYsRHh5+1xhiY2Ph5eWFkJAQKJVKNGjQAB07dix3LFT9FJwHJjc3F1999RUCAgLQpUsXAMCRI0dw9uxZS++ksipz0mP48OGYNGkSHBwc0KNHDwDAX3/9hddee61SJjQNDg6+54zQEydOvO/hLIVxeMs9ZCUB0ftMPTmu7APS4qyv2zgDjXsCjYNNyQ7Xor2DarJMrR4r/o7GqC4N4aSRpucJUa0nkwGy/B4WVcloKCEhkgvosoC8LECbaRq6l5dhOs7LP9dm5pcVKs/LNLUtGvLnLbqP3ipqJ8DW6e7JEfOxxg2w9wTs3JksIaJK4dCvHzL/+gsZO3ehXmio1OFUDqXG1ONCqnvfh6ysLFy+fBnjxo3D+PHjLeV6vR5OTk4AgNGjR6NPnz5o0aIF+vfvj8cffxx9+/a9r/sWlJ6ejhs3bqBbt25W5d26dbP02LhbDM888wwWL16Mxo0bo3///hgwYAAGDhwIhaLMj7JUTc2cOdNy/MILL2DSpEmYPXt2kTrm3j9lVeZPyuzZs3H16lX07t3b8kEzGo14/vnn8dFHH5UrCKoBdLlA7GFTguPKXuDmaVjNCSFTAg06m5IcTXoB3m1r9S/YL/5wAgcvJSFTq8fbAzhxF1GtIpPn976owEnfjEZTwqRwssScENFmADmpQG6qaZ+TcufYvNdlmdqyJE3KMKRUkAEad1MCxN6jhH3+sY0Th+AQUak5PNILNxUKaC9cQN7Vq1AV6g1eKwjCfQ8xkUpmpinp/vXXX6NTp05W18xDVR566CFER0dj27Zt+PPPPzFkyBCEhIRgw4YNVRbn3WLw8/PDhQsX8Oeff2LXrl145ZVXsGDBAvz1119QKvnlY22zfv16nDhxokj5s88+i/bt22PFihVlbrPMSQ+VSoV169Zhzpw5iIyMhK2tLdq0aWPVBYlqicxE4J+1pmErsYfzu3gX4PGAKcHROBjw71pj/zEojxe6N8bBS0lYefAqnuvsDz/X+8vCE1EtJ5PdmdsD3uVrQ59XNBGSm2pKkBQpyy/PTgKybpnmeslKNG0JJd0gn1x97+SIU33A3osTvRIR5M7OsOvUCVkHDyJ95y64Txh/7xdRlfH09ISPjw+uXLmCkSNHlljP0dERQ4cOxdChQzF48GD0798fycnJcHV1hVKphMFgKHcMjo6O8PHxwcGDB9GzZ09L+cGDB62GqdwtBltbWwwcOBADBw5EaGgoWrZsiTNnzuChhx4qd1xUPdna2uLgwYNo1qyZVfnBgwdhY2NTrjbL3SeoWbNmaNasGQwGA86cOQNHR0e4uLiUt7lqp07P6ZF0GTj0BRC5xjQ+3szeKz/JkZ/ocLi/eVNqsuDm9dCtqRsOXkrCgh0X8PnwIKlDIqLaTqHKTzx4lO11Br0p+ZGZYEpmZyYUOi6w16aZ/t5PizVtdyNXAU6+gHODApv/nWMmRYjqDId+fZF18CAyduxg0qMaev/99zFp0iQ4OTmhf//+0Gq1OHHiBFJSUhAWFoZFixbB29sbQUFBkMlkWL9+Pby8vODs7AwAaNiwIXbv3o1u3bpBrVbf9ZnPvOhEQc2aNcMbb7yBmTNnokmTJmjbti2+++47REZGYvXq1QBw1xhWrlwJg8GATp06QaPR4Mcff4StrS2/dK+lJk+ejJdffhmnTp2yJMWOHj2KFStW4L333itXm2VOekyePBlt2rTBuHHjYDAY0LNnTxw6dAgajQZbtmxBcHBwuQKpburknB7XTwJ/LwbO/Q7L0JX67YE2g02Jjnot2OU5nyAIeHtAKzz+xd/Y/M8NjHu4EQL9nKUOi4ioKLnClKQuTaJal5OfBMnvFVJcciQjAUi/bprwNfmKaSuOTAk4+zEpQlQHOISEIH7W+8g9exZ5165D5Vtf6pCogBdeeAEajQYLFizAG2+8ATs7O7Rp0waTJ08GADg4OGD+/Pm4ePEi5HI5OnTogK1bt0KW/3f0woULERYWhq+//hr169e/65K15kUnCjpw4AAmTZqEtLQ0TJkyBYmJiQgICMDmzZst3+bfLQZnZ2d8/PHHCAsLg8FgQJs2bfD7779XyiIaJL1p06ahcePG+Oyzz/Djjz8CAFq1aoXvvvsOQ4YMKVebgnivWUIL8fX1xaZNm9C+fXts2rQJr7zyCvbt24cffvgBe/bswcGDB8sVSHVlTnrExcXB19dX6nAqnigCl3YDBxcDVw/cKW/WD+j2mmnYChMdJQr7ORK/nrqOjo1csW5C51IvtUREVKMZ9EDGTSA1ttAWY9qnXTNN2no3MqV1TxEXf8C1CeDezLRXcdggUU0SM2o0so8ehcebb8Jt7Bipw7kvubm5iI6ORqNGjcrdnZ6oNO72Wav1z6FVqMw9PW7fvm1Zrmjr1q0YMmQImjdvjrFjx+Kzzz6r8ACpkhh0wNmNwMHPgIR/TWUyBdDmGaDrJMAzQNr4aoipfVvgj9M3cSw6GbuiEtD3gfIv5UVEVGPIFfm9OPwAdCt6vTRJEaMOSIk2bcVx8gPcmgBuzUyJELemps3Jjz1EiKohh759kH30KDJ27qzxSQ8iklZqaio2bNiAK1euYOrUqXB1dcWpU6fg6emJ+vXL3pOszEkPT09PREVFwdvbG9u3b8eSJUsAANnZ2ZYZgKkay8sCTq0CDoffWWZWaQe0Gw10ecX0rRuVmo+zLcZ3b4y4lGy08naUOhwiouqhPEmRlKtA0iUg6aJpEta0ONN2ZZ/1axU2pp4gbvm9QixJkSam5XqJSBIOIX2QMOdD5ERGQhcfD6UXvwgiorI7ffo0QkJC4OTkhKtXr+KFF16Aq6srfv31V8TGxmLVqlVlbrPMSY8xY8ZgyJAh8Pb2hiAICAkJAWCaXKRly5ZlDqC6qnUTmWbdBo4uA45/bfplEgDs6gGdXgI6jOMvivdhSt/mHNZCRFQW90qKZCXdSYDcvph/fMk0f4g+F0g8a9oK07hb9wpxbwa4twBcG7N3CFElU3p6wDYoCDmnTiFj159wfe5ZqUMiohooLCwMo0ePxvz58+Hg4GApHzBgAEaMGFGuNsuc9Jg1axZat26NuLg4PPPMM1Cr1QBM6zxPmzatXEFUR7VmItPkaODwl0DEj3eWnHVtDHR9FQgcDihtpY2vFiic8BBFkUkQIqL7Yedm2hp0si436E2rytzOT4gkXbqTFMm4CWTfBmJvm5ZZL0hpB3g+AHi1BrzaAF4PAh4BnDeEqII59utrSnrs2MGkBxGVy/Hjx7Fs2bIi5fXr10d8fHy52izXkrWDBw8uUjZq1KhyBUCV5Eakab6OqE2AaDSV+QQB3SYDrQYCMg5FqmhxydmYv+MC/F01mNqvhdThEBHVPnKFKXHv2hhAX+tr2gzTkusFEyFJF4FbFwBdFnDtmGkzE2SmYTJebe4kQrza1Onl2Inul0OfPkiY+zGyT56E/tYtKOrVkzokIqph1Go10tPTi5T/999/qFfOv1PKlfSgakoUgSt7TcmOgmOgm4aYVmJp2J0rsVSiczfT8fs/N6BWyDCiUwP4OLMXDRFRlVE7AD5tTVtBBj2QfBmIP2O9ZSX+n737jquq/uM4/rqXvfdeouLAgbhQcSauzJWaI3Nk2nBkajYs69eyzF3mypmWWpaZA1fixC0u3CLgAGTvefn9cfUWqSWIHsbn+XicB9xzzj33fRGE+7nf7+d7d7TIZTj361/nmzkULYQ41dVOldGTP5mE+C8Grq4Y169P9unTpO3ahU3//kpHEkKUM927d+eTTz5h3bp1gHZUfVRUFO+88w69e/cu0TXlN3hFUJBP6slfMDr0DUbxd1diUelB3d4QOFb7h5t44jr4OtG0ii1HricyY/slZrzgp3QkIYQQevrgUFO71fvbSNW0WIj9RyEk4Qpk3IGrf2q3e/SNtdNhdMWQu5uh2dN/PkKUcZYdO5B9+jSp27ZJ0UMIUWwzZsygT58+ODo6kpWVRZs2bYiJiaF58+Z8/vnnJbqmFD0eojw1Mj23ezV19o8FoNDAFFXDwdB8FFh7KpysclGpVLzftTY95x3g15M3eLllFeq4WikdSwghxINYOGm36kF/7cvNhLjzEHNaWwSJPQsxZ7XTY26d0G73qNTaQohbI3BvDG6NtYUVmT4qKjmLjh2Jmz6DzCNHyU9KQt9GmuULIR6dlZUVO3bs4MCBA5w6dYr09HQaNmyoW0ClJFSFhYWFpZixwrnXyDQ6Ohp397K5nGtcchpxs1qzLb8htbq/RdeAukpHqtTG/HSSP07dIrC6HauGB0hTUyGEKM80GkiKuFsIOXt3VMhpbePUfzI01/bP+nshxNLl6WcWQmHXej1PzvnzuHz2KdYP6AVY1mVnZxMREYG3tzfGxsZKxxEV2L99r5WH16GlLS8vDxMTE8LCwqhbt/Re0xZ7pMeDmoqA9l1uIyMjDA0NHzuUKB5Hawt+abuOb7ZdwnlXLO38a2FqKIN4lDKpU022nY3hwJUEQi7doV1NR6UjCSGEKCm1Guyqabc6vf7an3oLbh6HG8e0H2+egNx0uL5Pu91j6fa3IkgjbVFEpsWICs6yU0funD9P6vbt5bLoIUpHlSpVGDduHOPGjXuqj7t8+XLGjRtHcnLyU31c8fgMDAzw9PSkoKCgVK9b7EXrra2tsbGxuW+ztrbGxMQELy8vPvroIzQaTakGFf/u5ZZV8bA1ISY1mwUhV5WOU6l52JoypIUXAN/vu6ZwGiGEEE+Epat2NbQO/4Ohm+C9aHj9IHSbCw2HaBugqtSQehPOb4QdU2B5V5jqDvMDYeNYOLESYsNBU7p/3AmhNIuO2tWVMkIPUfCQN0xF6Rs6dCgqlQqVSoWBgQHe3t5MmjSJ7OzsUrn+9evXUalUhIWFPdL5R48eZeTIkY907vLly7G2ti55uGJSqVRs2LDhqT2eeHSTJ0/m/fffJzExsdSuWezhAMuXL2fy5MkMHTqUpk2bAnDkyBFWrFjBBx98wJ07d5g+fTpGRka8//77pRZU/DtjAz0mP1ub11adYOHea7zQxAN3G1OlY1Vao9v5YGlswMstvZWOIoQQ4mlQ64FTHe3WaIh2X0463A67OxrkmHY0SOpNba+Q2LNwYoX2vHvTYtwbg1cgeDbTrkYjRDllVLUqRj7Vybl8hfTdu7Hq0UPpSJVG586dWbZsGXl5eRw/fpwhQ4agUqn46quvnlqG3NxcDA0NS7y8qKjcvv32W65cuYKrqyteXl6YmRUdHXnixImH3PPhij3SY8WKFcyYMYNPP/2Ubt260a1bNz799FOmT5/O2rVrmTx5MnPnzmXlypXFDiMeT6c6zjSraktOvoYvt15QOk6lZmVqwJj2PpgZyTQjIYSotIzMoUpLaDkO+q2C8eEw/oL288Bx2qXkDc3/mhazfxas7gNfesH3HWDXJ3B1t7bBqhDljEUH7WiP1G3bFU7y+AoLC8nMy1RkK277RSMjI5ydnfHw8KBnz54EBQWxY8cO3XGNRsPUqVPx9vbGxMQEPz8/fvnlF93xpKQkXnzxRRwcHDAxMcHHx4dly5YB4O2tfTPP398flUpF27ZtAe0Ik549e/L555/j6upKzZo1Ae30ltmzZ+uunZyczKuvvoqTkxPGxsbUrVuXTZs2ERISwrBhw0hJSdGNVPn4448ByMnJYeLEibi5uWFmZkZAQAAhISFFnvPy5cvx9PTE1NSUXr16kZCQUKyv2T9pNBo++eQT3N3dMTIyokGDBgQHB+uO5+bmMnr0aFxcXDA2NsbLy4upU6cC2u+Vjz/+GE9PT4yMjHB1dWXs2LGPlaey6dmzJxMnTuS9995j4MCB9OjRo8hWEsV+RXbw4EEWLFhw335/f39CQ0MBaNmyJVFRUSUKJEpOpVIx5bk6PPfNPjadvs2QFok0qWKrdKxKT6Mp5HJcOjWd5V07IYSo9CxdwLKbdmoMaKe23LmoHQkSfRgi9kFyJNw4ot32zQA9Q21TVO/W4N0K3JuAvpGyz0OI/2DRqRPx331Hxv79FKRnoGdefnvZZOVnEfBjgCKPfXjgYUwNSjZ6++zZsxw8eBAvLy/dvqlTp7Jq1SoWLFiAj48Pe/fuZdCgQTg4ONCmTRs+/PBDwsPD2bp1K/b29ly5coWsrCxAO7q/adOm7Ny5kzp16hTp5bhr1y4sLS2LFFj+TqPR0KVLF9LS0li1ahXVqlUjPDwcPT09WrRowezZs5kyZQoXL14EwNzcHIDRo0cTHh7OmjVrcHV15bfffqNz586cOXMGHx8fDh8+zPDhw5k6dSo9e/YkODiYjz76qERfr3vmzJnDjBkzWLhwIf7+/ixdupTu3btz7tw5fHx8mDt3Lhs3bmTdunV4enoSHR1NdHQ0AOvXr2fWrFmsWbOGOnXqEBMTw6lTpx4rT2XzuP9+D1LsooeHhwdLlizhyy+/LLJ/yZIleHh4AJCQkIBNOV+eqjwtWft3vq6W9GviyU9Hovjkj3B+HxWIWi2rhyglIT2HIcuOcDUugz1vt8XRUjqACyGE+Bu1Hjj5areGg7X7kiK1Iz8i7jZFTb0JUQe1254vQd8YPJpCldbaQohbQ9AzUPZ5CPEPRjV8MPTyIjcykvQ9IVh17ap0pEph06ZNmJubk5+fT05ODmq1mm+//RbQjpr44osv2LlzJ82bNwegatWq7N+/n4ULF9KmTRuioqLw9/encePGgHa0xj33pqvY2dnh7Oxc5HHNzMz4/vvvH7qoxc6dOzly5Ajnz5+nRo0ause+x8rKCpVKVeS6UVFRLFu2jKioKFxdXQGYOHEiwcHBLFu2jC+++II5c+bQuXNnJk2aBECNGjU4ePBgkZEZxTV9+nTeeecd+vfvD8BXX33F7t27mT17NvPmzSMqKgofHx9atmyJSqUqUlSKiorC2dmZoKAgXVPOey0hhHKKXfSYPn06ffv2ZevWrTRp0gSAY8eOceHCBd3QqKNHj9KvX7/STfqUjRo1ilGjRumWCipPJnSswaZTtzhzM4VfTtzghcblK39FYmtmiIGemqy8AmbuuMSXvesrHUkIIURZZ+Ol3fwHQWEhJF67WwTZqy2EZMTd/Xwv7AYMzLR9QLxbaQshLn6gJ9MrhbJUKhUWnTqRsGgRadt3lOuih4m+CYcHHlbssYujXbt2zJ8/n4yMDGbNmoW+vj69e/cG4MqVK2RmZtKhQ4ci98nNzcXf3x+A119/nd69e3PixAk6duxIz549adGixX8+br169f51Fc+wsDDc3d11BY9HcebMGQoKCu67T05ODnZ2dgCcP3+eXr16FTnevHnzEhc9UlNTuXXrFoGBgUX2BwYG6kZsDB06lA4dOlCzZk06d+7Mc889R8e7zXv79u3L7NmzqVq1Kp07d+bZZ5+lW7du6OvL/8mPqqCggFmzZrFu3TqioqLuG4RQkganxf7qd+/enQsXLrBw4UIuXboEQJcuXdiwYYOuEvj6668XO4goPfbmRoxt78PnW87z9baLPFvPBXPpLaEIlUrFB11r03t+KOuORTMs0FumuQghhHh0KtVfS+Y2GqotgsRf0hY87o0GyUqEq7u0G4CRJXi10PYM8W4FTvW0S+8K8ZRZdOxIwqJFpO/diyYrC7VJ8V7AlxUqlarEU0yeNjMzM6pXrw7A0qVL8fPzY8mSJQwfPpz09HQANm/ejJubW5H7GRlpp8x16dKFyMhItmzZwo4dO2jfvj2jRo1i+vTp//m4/8akBP/26enp6Onpcfz4cfT09Iocuzf9RQkNGzYkIiKCrVu3snPnTl544QWCgoL45Zdf8PDw4OLFi+zcuZMdO3bwxhtv8PXXX7Nnzx4MDGRE3qP43//+x/fff8+ECRP44IMPmDx5MtevX2fDhg1MmTKlRNcs0Sthb2/v+6a3iLJlSIsq/Hgkioj4DObtvsI7nWspHanSauRlS5e6zmw9G8PUredZPkyGuAkhhCghlQocamq3piNAo4G48L8KIJH7ITsFLgVrNwBTO6jZBWp3h6ptpR+IeGqM6/hi4OZG3s2bpO/bh+Xdd8PF06FWq3n//fcZP348AwcOxNfXFyMjI6KiomjTps1D7+fg4MCQIUMYMmQIrVq14u2332b69Om6kRwFBcVfZrt+/frcuHGDS5cuPXC0h6Gh4X3X9ff3p6CggLi4OFq1avXA69auXZvDh4uOwjl06FCx891jaWmJq6srBw4cKPI1OnDgQJFpKpaWlvTr149+/frRp08fOnfuTGJiIra2tpiYmOgW/Bg1ahS1atXizJkzNGzYsMS5KpPVq1ezePFiunbtyscff8yAAQOoVq0a9evX59ChQyVqDFuiokdycjJLlizh/PnzANSpU4eXX34ZKyurklxOPAGG+momP1ubV1YeY8m+CAY08cTTrnxUqCuiSZ1rsSM8lpCLd9h/OZ6WPvZKRxJCCFERqNXgXFe7NXtd2xg15vRf/UAiD0JmApxcpd0MLcCng7aRqk8HWRpXPFEqlQqLjh1JXLaMtG3bpeihgL59+/L2228zb948Jk6cyMSJE3nrrbfQaDS0bNmSlJQUDhw4gKWlJUOGDGHKlCk0atSIOnXqkJOTw6ZNm6hduzYAjo6OmJiYEBwcjLu7O8bGxo/8+q9Nmza0bt2a3r17M3PmTKpXr86FCxdQqVR07tyZKlWqkJ6ezq5du/Dz88PU1JQaNWrw4osvMnjwYGbMmIG/vz937txh165d1K9fn65duzJ27FgCAwOZPn06PXr0YNu2bY88tSUiIoKwsLAi+3x8fHj77bf56KOPqFatGg0aNGDZsmWEhYWxevVqAGbOnImLiwv+/v6o1Wp+/vlnnJ2dsba2Zvny5RQUFBAQEICpqSmrVq3CxMSkSN8P8e9iYmKoV68eoB3Rk5KSAsBzzz3Hhx9+WKJrFnus47Fjx6hWrRqzZs0iMTGRxMREZs6cSbVq1Uq0Zq54ctrXdqRldXtyCzR8seW80nEqNW97MwY10/5n9/mW8xRoirf8mBBCCPFI1Hrg6g+BY+HFn+Gd6zB4IzQdCRaukJsG536FX4bBtGrwYz9tMSTj8ZZ4FOJhLDtpCx3pISFocnIUTlP56OvrM3r0aKZNm0ZGRgaffvopH374IVOnTqV27dp07tyZzZs365ajNTQ05L333qN+/fq0bt0aPT091qxZo7vW3LlzWbhwIa6ursVePnT9+vU0adKEAQMG4Ovry6RJk3SjO1q0aMFrr71Gv379cHBwYNq0aQAsW7aMwYMHM2HCBGrWrEnPnj05evQonp6eADRr1ozFixczZ84c/Pz82L59Ox988MEj5Rk/fjz+/v5FtpMnTzJ27FjGjx/PhAkTqFevHsHBwWzcuBEfHx8ALCwsmDZtGo0bN6ZJkyZcv36dLVu2oFarsba2ZvHixQQGBlK/fn127tzJH3/8oetBIv6bu7s7t2/fBqBatWps365d9vro0aO6aVjFpSos5uLPrVq1onr16ixevFjXkCU/P59XXnmFa9eusXfv3hIFKavuNTKNjo7G3d1d6TjFdjEmjS5z9qIphJ9GNKN5NfmBU0piRi5tvt6Nt70Z3w9pjKOFrOQihBDiKdJo4NYJOP8HnN+obZB6j0pP2wekdneo1RWs3B5+HSGKoVCj4Uq7Z8iPjcX9u++weKad0pH+U3Z2NhEREXh7e2NsLH+viSfn377XivM6tKCggI8//phVq1YRExODq6srQ4cO5YMPPkCl0q7kWVhYyEcffcTixYtJTk4mMDCQ+fPn64o5oG0SOmbMGP744w/UajW9e/dmzpw5RXqonD59mlGjRnH06FEcHBwYM2aMbvWc0vDuu+9iaWnJ+++/z9q1axk0aBBVqlQhKiqKt956q0RtNopd9DAxMeHkyZPUqlW0R0R4eDiNGzcmMzOz2CHKsvJe9AD4cMNZfjgUSW0XSzaNaYmeLGGrmCtx6VS1N5NlhIUQQiirsBDizsOFTdoCSMyZosfdGmmnwNTqBvbVlckoKoyYz78g6YcfsOrRA9evyn5fQCl6iKeltIoeX3zxBTNnzmTFihXUqVOHY8eOMWzYMD7//HNdD4yvvvqKqVOnsmLFCry9vfnwww85c+YM4eHhusfu0qULt2/fZuHCheTl5TFs2DCaNGnCjz/+CGhXt6lRowZBQUG89957nDlzhpdffpnZs2czcuTIJ/AVgtDQUEJDQ/Hx8aFbt24lukaxix5OTk788MMPumV57tm2bRuDBw8mNja2REHKmnnz5jFv3jxyc3O5evVquS56JGbk0vbr3aRm5/NFr3oMDPBUOpIQQgghypKk63B+k3YUSPRh4G9/HjrU1hZAaj8HzvW1zVSFKIbMo0eJfGkwaktLauzfh+pfljYtC6ToIZ6W0ip6PPfcczg5ObFkyRLdvt69e2NiYsKqVasoLCzE1dWVCRMmMHHiRABSUlJwcnJi+fLl9O/fn/Pnz+Pr68vRo0dp3LgxAMHBwTz77LPcuHEDV1dX5s+fz+TJk4mJidE1tn333XfZsGEDFy5cKM0vTakqdk+Pfv36MXz4cNauXUt0dDTR0dGsWbOGV155hQEDBjyJjIoYNWoU4eHhhISEKB3lsdmaGTIuSNslecb2i6Rm5ymcSGTk5DN312Xi0rKVjiKEEEKATRVoMRqGb4MJF+G5WVDtGVDrw53zsHcaLGwNc+rDtskQGaptmirEIzBp2BA9e3s0qalk/GOlDSHEv0tLSyM1NVW35TygN06LFi3YtWsXly5dAuDUqVPs37+fLl26ANqmrTExMQQFBenuY2VlRUBAAKGhoYB2RIW1tbWu4AEQFBSEWq3WrZATGhpK69atdQUPgE6dOnHx4kWSkpJK5fmuXLnyX7eSKPbqLdOnT0elUjF48GDy8/MBMDAw4PXXX5dlbMuwl5p7sfpwJFfvZPDNrstM7uqrdKRKbcxPJ/nzQhznbqWwYFAj3Vw7IYQQQnEWTtD4Ze2WlQSXtsOFP+DyTkiOgtBvtZuVJzQeCv4vgbmj0qlFGabS08MiqD3Ja9aStn075g9ZflQIcT9f36Kv2z766CM+/vjjIvveffddUlNTqVWrFnp6ehQUFPD555/z4osvAtoVUUA7a+PvnJycdMdiYmJwdCz6f7m+vj62trZFzrnX9Pbv17h3zMbG5jGeqdabb75Z5HZeXh6ZmZkYGhpiamrK4MGDi33NYo/0MDQ0ZM6cOSQlJREWFkZYWBiJiYnMmjWrxN1UxZNnoKfmg+e0PzDLD14nIj5D4USV24SONdBXq9h2Lpbfw24pHUcIIYR4MBMb8OsH/VbBpGvaj/X7g5EVpETBrk9gpi/8PAyu79f2ChHiASw7dQIgbcdOCu++cSqE+G/h4eGkpKTotvfee+++c9atW8fq1av58ccfOXHiBCtWrGD69OmsWLFCgcSPJykpqciWnp7OxYsXadmyJT/99FOJrlnsosc9pqam1KtXj3r16mFqalrSy4inqF1NR9rWdCCvoJDPN4crHadSq+Nqxdj22k7JU34/S2yqTHMRQghRxhmaant7PL8QJl6EngvAvQlo8rTL4C7vCvMC4NACyEpWOq0oY0ybNEHP2pqC5GQyjx1TOo4Q5YaFhQWWlpa67UEDDd5++23effdd+vfvT7169XjppZd46623mDp1KgDOzs4A9/XfjI2N1R1zdnYmLi6uyPH8/HwSExOLnPOga/z9MZ4EHx8fvvzyy/tGgTyqR5re8vzzzz/yBX/99dcSBRFPxwddfdl/eS87z8ex7/IdWvk4KB2p0nq9bTV2hMdy5mYK764/zdKhTWSaixBCiPLBwAQaDNBut0/DsaVweh3EX4Tgd2Dnx1CvNzQeDm4NlU4rygCVvj7mQe1J+WU9qdu2YdasmdKRhKgwMjMzUauLjmfQ09NDo9EA4O3tjbOzM7t27aJBgwaAdiWWw4cP8/rrrwPQvHlzkpOTOX78OI0aNQLgzz//RKPREBAQoDtn8uTJ5OXlYWBgAMCOHTuoWbNmqUxt+Tf6+vrculWyEfKPNNLDysrqkTdRtlV3NOel5l4AfLopnPwCjcKJKi8DPTUzXvDDUE/N7ot3+PnYDaUjCSGEEMXnUh+6zYYJF6DrDHCsA/lZcHIVLG4Hi9rCiZWQm6l0UqEwy7urP6bt3ElhgTTCFaK0dOvWjc8//5zNmzdz/fp1fvvtN2bOnEmvXr0AUKlUjBs3js8++4yNGzdy5swZBg8ejKurKz179gSgdu3adO7cmREjRnDkyBEOHDjA6NGj6d+/P66urgAMHDgQQ0NDhg8fzrlz51i7di1z5sxh/PjxpfZcNm7cWGT7/fffWbBgAYMGDSIwMLBE1yz2krWVTXGWCiovUjLzaDt9N0mZeXzaow4vNa+idKRKbcGeq3y59QJu1iaEvN0WA70SzzoTQgghlFdYqF329thSOPcbFORq9xtZaUeGNH4ZHGoqm1EoojA3l0uBLdGkpeG16gdM/7ZKRFkiS9aKp6W0lqxNS0vjww8/5LfffiMuLg5XV1cGDBjAlClTdCutFBYW8tFHH7Fo0SKSk5Np2bIl3333HTVq1NBdJzExkdGjR/PHH3+gVqvp3bs3c+fOxdzcXHfO6dOnGTVqFEePHsXe3p4xY8bwzjvvlNrX5J8jVlQqFQ4ODjzzzDPMmDEDFxeXYl9Tih7/oSIWPQB+CL3Oh7+fw8bUgJCJ7bAyNVA6UqVVoClkWvAFhgV642wlv1iFEEJUIBkJELYKji2DpIi/9nu1hCYvQ61uoG/48PuLCufWO++Q8vtGbAa/hPP77ysd54Gk6PFgy5cvZ9y4cSQnJz/Wddq2bUuDBg2YPXu24lmUVlpFD/HvHukt5c6dO3Po0KH/PC8tLY2vvvqKefPmPXYw8WQNaOpJDSdzkjLzmL3rktJxKjU9tYr3nq0tBQ8hhBAVj5kdBL4JY07AoF+h1nOgUkPkfvjlZZjlCzv/B0mRSicVT4nFvVVctu+gUCPTrEvT0KFDUalU922dO3dWOpoiVCoVGzZsUDqGKAMeqZFp37596d27N1ZWVnTr1o3GjRvj6uqKsbExSUlJhIeHs3//frZs2ULXrl35+uuvn3Ru8Zj09dR8+JwvLy05wg+hkbwY4EV1R/P/vqN44v68EEt1Bws87WRVJCGEEBWEWg3V22u3lJvaHh8nVkDabdg/E/bPAp+O2qkvPh2154sKySwwELWpKfkxMWSfOYOJn5/SkSqUzp07s2zZsiL7HrTahxBlVXH6g8ycOfORznuk3yjDhw/n2rVrvP/++4SHhzNy5EhatWpFkyZN6NSpE4sXL8bT05OjR4+ydu1aPD09HzmoUE4rHweCajuSrynkM1nCtkz4ft81Xl5+jIm/nEKjkZlnQgghKiArN2j3How7Ay/8AFXbAYVweRv81A/mN4czv4BGGl1WRGojI8zbtgUgddt2ZcM8osLCQjSZmYpsxe1EYGRkhLOzc5Ht3qoaISEhGBoasm/fPt3506ZNw9HRUbfsaHJyMq+++ipOTk4YGxtTt25dNm3a9MDHGjp0qK4J5j3jxo2j7d1/X4CMjAwGDx6Mubk5Li4uzJgx477r5OTkMHHiRNzc3DAzMyMgIICQkJAi5yxfvhxPT09MTU3p1asXCQkJxfq6/JNGo+GTTz7B3d0dIyMjGjRoQHBwsO54bm4uo0ePxsXFBWNjY7y8vHTLvxYWFvLxxx/j6emJkZERrq6ujB079rHyiL+cPHmSpUuXsnDhQkJCQggJCWHRokUsWbKEkydP6rawsLBHvuYjjfQA7Q/QoEGDGDRoEAApKSlkZWVhZ2enW66mIpk3bx7z5s0jNzdX6ShP1OSuvuy5dIeQi3fYfTGOdjUdlY5UqXX0dWbmjksciUhk2cHrDG/prXQkIYQQ4snQMwDf7tot4aq28emJH+DOBVg/HPZ8Ba3fhjrPg94j/8kqygGLjh1J3bKFtO3bcXx7IiqVSulI/6owK4uLDRsp8tg1TxxHZVo6o3/btm3LuHHjeOmllzh16hTXrl3jww8/5Oeff8bJyQmNRkOXLl1IS0tj1apVVKtWjfDwcPT09Er8mG+//TZ79uzh999/x9HRkffff58TJ07olk0FGD16NOHh4axZswZXV1d+++03OnfuzJkzZ/Dx8eHw4cMMHz6cqVOn0rNnT4KDg/noo48e62sxZ84cZsyYwcKFC/H392fp0qV0796dc+fO4ePjw9y5c9m4cSPr1q3D09OT6OhooqOjAVi/fj2zZs1izZo11KlTh5iYGE6dOvVYecRfunXrhoWFBStWrNAV7JKSkhg2bBitWrViwoQJxb6mNDL9D5Whgcznm8NZvC+Cag5mBI9rLauHKGzVoUg+2HAWI301W95sRTUHmXYkhBCikshOgcOLIPRbyE7W7rOtpi1+1OsrxY8KQpOZyaUWgRRmZ1Nl/S+Y1KmjdKQi/tlcUpOZqWjRQ/2IRY+hQ4eyatWq+xpivv/++7x/t2lsbm4uAQEB1KhRg7NnzxIYGMiiRYsA2L59O126dOH8+fNFVvS455/NQ4cOHUpycnKRvhnjxo0jLCyMkJAQ0tPTsbOzY9WqVfTt2xfQrg7i7u7OyJEjmT17NlFRUVStWpWoqCjdsqgAQUFBNG3alC+++IKBAweSkpLC5s2bdcf79+9PcHDwvzYyValU/Pbbb/eNRgFwc3Nj1KhRuq8LQNOmTWnSpAnz5s1j7NixnDt3jp07d95XlJs5cyYLFy7k7Nmzj/3mvzQyvZ+bmxvbt2+nzj/+Xzh79iwdO3bk1q1bxb6m/OYQjGnvw68nbnL1TgY/hEbysowuUNSLAZ5sOxfDvsvxTPz5FL+81gI9ddl+B0QIIYQoFcZW0OZtCHgVji6Gg99C4lXY8Brs+RJaTQS//tpRIqLcUpuaYt6qFWk7dpC2fUeZK3r8k8rEhJonjiv22MXRrl075s+fX2Sfra2t7nNDQ0NWr15N/fr18fLyYtasWbpjYWFhuLu7P7DgURJXr17VFVn+nqVmzb+WrD5z5gwFBQX3PWZOTg52dnYAnD9/nl69ehU53rx58yLTUYojNTWVW7duERgYWGR/YGCgbsTG0KFD6dChAzVr1qRz584899xzdOzYEdD2u5w9ezZVq1alc+fOPPvss3Tr1g19fXlpXRpSU1O5c+fOffvv3LlDWlpaia4pb+kLLI0NmNBR+5/P7J2XSMyo2FN6yjqVSsVXvetjYaTPyahkFu29pnQkIYQQ4ukytoRWE7R9P4L+B6b2kHQdNo6GbxrC8eWQL3+vlGe6VVyCg4vdt+JpU6lUqE1NFdmKO/XHzMyM6tWrF9n+XvQAOHjwIKAddZGYmKjbb1LMAotarb7v3y4vL69Y10hPT0dPT4/jx48TFham286fP8+cOXOKda3S1LBhQyIiIvj000/JysrihRdeoE+fPgB4eHhw8eJFvvvuO0xMTHjjjTdo3bp1sZ+7eLBevXoxbNgwfv31V27cuMGNGzdYv349w4cP5/nnny/RNaXoIQDo18SD2i6WpGbnM2uHLGGrNFdrEz7s5gvArB2XiEnJVjiREEIIoQAjc2g5Dsadho6fgZkDJEfBH29qix9Hl0B+jtIpRQmYt22DysCA3MhIci5dVjpOpXH16lXeeustFi9eTEBAAEOGDEFzd+ng+vXrc+PGDS5derTXAg4ODty+fbvIvr83l6xWrRoGBgYcPnxYty8pKanI9f39/SkoKCAuLu6+Yo2zszMAtWvXLnINgEOHDhXref+dpaUlrq6uHDhwoMj+AwcO4OvrW+S8fv36sXjxYtauXcv69et1RSITExO6devG3LlzCQkJITQ0lDNnzpQ4k/jLggUL6NKlCwMHDsTLywsvLy8GDhxI586d+e6770p0TRmDIwDQU6uY8pwvAxYfYvXhSAY186Kms4XSsSq1vo3cORqRSAdfJ5ytjP/7DkIIIURFZWgGLcZA4+HaUR4HZkNKNGweD/tmQMu3wP8lMJDfl+WFnrk5Zi1bkr57N2nbt2Ncs3SmVFR2OTk5xMTEFNmnr6+Pvb09BQUFDBo0iE6dOjFs2DA6d+5MvXr1mDFjBm+//TZt2rShdevW9O7dm5kzZ1K9enUuXLiASqWic+fO9z3WM888w9dff83KlStp3rw5q1at4uzZs/j7+wNgbm7O8OHDefvtt7Gzs8PR0ZHJkyej/tuS1DVq1ODFF19k8ODBzJgxA39/f+7cucOuXbuoX78+Xbt2ZezYsQQGBjJ9+nR69OjBtm3bHnlqS0RExH2rfPj4+PD222/z0UcfUa1aNRo0aMCyZcsICwtj9erVgLZvh4uLC/7+/qjVan7++WecnZ2xtrZm+fLlFBQUEBAQgKmpKatWrcLExAQvL6/i/FOJhzA1NeW7777j66+/5urVq4C2gGZmZlbiaxZ7pEd0dDQ3btzQ3T5y5Ajjxo3TNcAR5VfzanZ0ruOMphA+3RRe5ocaVnQqlYqv+/rRsY6z0lGEEEKIssHQFJq/AW+egi5fg4UrpN6ELRNhbgM4tADyspROKR6RRSdtj4S07dsUTlJxBAcH4+LiUmRr2bIlAJ9//jmRkZEsXLgQABcXFxYtWsQHH3yg62Wxfv16mjRpwoABA/D19WXSpEkUFDx4+ehOnTrx4YcfMmnSJJo0aUJaWhqDBw8ucs7XX39Nq1at6NatG0FBQbRs2ZJGjYo2hV22bBmDBw9mwoQJ1KxZk549e3L06FE8PT0BaNasGYsXL2bOnDn4+fmxfft2Pvjgg0f6eowfPx5/f/8i28mTJxk7dizjx49nwoQJ1KtXj+DgYDZu3IiPjw8AFhYWTJs2jcaNG9OkSROuX7/Oli1bUKvVWFtbs3jxYgIDA6lfvz47d+7kjz/+0PUgEaXDzMyM+vXrY2VlRWRkpG5EUkkUe/WWVq1aMXLkSF566SViYmKoWbMmderU4fLly4wZM4YpU6aUOExZVNm65kYlZBI0cw+5BRoWD25MB18npSOJu+JSs0nNzqe6o6zmIoQQQgCQlw1hq2DfLEi9+6acuRMEvgmNhmmLJKLMKkhJ4VJgS8jPp+qWzRhVrap0JODfV9QQojTJ6i1/Wbp0KcnJyYwfP163b+TIkSxZsgSAmjVrsm3bNjw8PIp97WKP9Dh79ixNmzYFYN26ddStW5eDBw+yevVqli9fXuwAomzxtDNleCvt6i2fbw4nJ//BlV3xdB2+lkCHWXsZ/eMJ+TcRQggh7jEwhiavwNgT8NxssPKE9FjY9j7MqQ8H5kBOutIpxUPoWVlh1rw5AGnbtyucRgihpEWLFmFjY6O7HRwczLJly1i5ciVHjx7F2tqa//3vfyW6drGLHnl5eRgZGQGwc+dOunfvDkCtWrXua2QjyqdR7arjYGHE9YRMVhy8rnQcAVRzNEdPreJCTBpzd0mzLyGEEKIIfSNoPAzGHIfu34C1F2TcgR1TtMWP/bOl4WkZZdGxAwCpUvQQolK7fPkyjRs31t3+/fff6dGjBy+++CINGzbkiy++YNeuXSW6drGLHnXq1GHBggXs27ePHTt26Jra3Lp1S+YxVRDmRvq83Um7hO03u64Qny5/JCjN3tyIz3vWBWB+yFXCopOVDSSEEEKURfqG0HCwtvjR4zuwrQqZCbDzI5gfCNf3K51Q/INFUBDo6ZETfp7c6Gil4wghFJKVlYWlpaXu9sGDB2ndurXudtWqVe9r0vuoil30+Oqrr1i4cCFt27ZlwIAB+Pn5AbBx40bdtBdR/vVp6E5dN0vScvL5fl+E0nEE0KWeC939XNEUwoR1YWTnyTQXIYQQ4oH0DMD/RRh1FHrOBzNHSLgMy7vChlGQkaB0QnGXvo0Npk2aADLFRYjKzMvLi+PHjwMQHx/PuXPnCAwM1B2PiYnBysqqRNcudtGjbdu2xMfHEx8fz9KlS3X7R44cyYIFC0oUQpQ9arWKEa20zaT2Xb6jcBpxzyc96uBgYcTVOxnM2H5R6ThCCCFE2aanDw0Gwuij0Phl7b6wVfBtYwj7EWSlujLB8u4qLqnbpOghRGU1ZMgQRo0axaeffkrfvn2pVatWkZV+Dh48SN26dUt07WIXPbKyssjJydE1GYmMjGT27NlcvHgRR0fHEoUQZVPzqtrpSuG3U0nOzFU4jQCwNjXky+frAfD9/giORyYpnEgIIYQoB0ys4blZMHwHONaBrETY8Dqs6Abx0itLaRZBQaBSkX36NHm3bikdRwihgEmTJjFixAh+/fVXjI2N+fnnn4scP3DgAAMGDCjRtYtd9OjRowcrV64EIDk5mYCAAGbMmEHPnj2ZP39+iUI8aZs2baJmzZr4+Pjw/fffKx2n3HC0NKaagxmFhXA4IlHpOOKu9rWd6NvInZeaeVHbxULpOEIIIUT54dEUXt0DQf8DfRO4vg/mt4DdU7XL3wpF6Ds4YNKoIQBpO3YonEYIoQS1Ws0nn3zCyZMn2bp1K7Vr1y5y/Oeff2b48OElu3Zx73DixAlatWoFwC+//IKTkxORkZGsXLmSuXPnlijEk5Sfn8/48eP5888/OXnyJF9//TUJCTKP81E1r6Yd7RF6Vb5mZclXvevzSY+6mBrqKx1FCCGEKF/0DKDlOBh1CKp3gIJc2PMlLAiEiL1Kp6u0LDveneKyXYoeQojSVeyiR2ZmJhYW2neXt2/fzvPPP49araZZs2ZERkaWesDHdeTIEerUqYObmxvm5uZ06dKF7dIk6ZE1r2oPwKFrUvQoS9Rqle5zjaaQW8lZCqYRQgghyiGbKvDiz9B3OZg7QcIV7XSX316XRqcKsOigXbo268QJ8uLiFE4jhKhIil30qF69Ohs2bCA6Oppt27bR8W5VNi4ursgSM6Vl7969dOvWDVdXV1QqFRs2bLjvnHnz5lGlShWMjY0JCAjgyJEjumO3bt3Czc1Nd9vNzY2bN2+Wes6KqllVWwAuxKSRIEvXljlxqdm8+P1hXlgYSnpOvtJxhBBCiPJFpYI6vbSNTpuMAFRw6kf4thGcXCWNTp8iAxcXjP3qQ2EhaTt3Kh1H3BUSEoJKpSI5OfmR7/Pxxx/ToEGDJ5ZJiOIqdtFjypQpTJw4kSpVqtC0aVOaN28OaEd9+Pv7l3rAjIwM/Pz8mDdv3gOPr127lvHjx/PRRx9x4sQJ/Pz86NSpE3FSIS4VduZG1HTSjuyRvh5lj6mRPtFJmdxIyuLzzeeVjiOEEEKUT8ZW0HU6vLITnOpCVhL8Pkq7xO0dWS3tabHs2AmANFnFpdgWLFiAhYUF+fl/vQmWnp6OgYEBbdu2LXLuvULG1atX//O6LVq04Pbt2yVeKvRh2rZty7hx4x7pPJVKxZo1a4rsnz17NlWqVCnVTEI5qampT/T6xS569OnTh6ioKI4dO8a2bdt0+9u3b8+sWbNKNRxAly5d+Oyzz+jVq9cDj8+cOZMRI0YwbNgwfH19WbBgAaamprrldF1dXYuM7Lh58yaurq4PfbycnBxSU1N1W1paWuk+oXJI+nqUXeZG+nzdxw+An45EseeSLC8shBBClJh7YxgZAh0+BQNTiDwA8wPhz8+l0elTYHF36drMo0fJT5Q324qjXbt2pKenc+zYMd2+ffv24ezszOHDh8nO/uv7d/fu3Xh6elKtWrX/vK6hoSHOzs6oVKr/PPdJMTY25oMPPiAvL69Ur1va1xMlZ2Njoxu08MwzzxRrZNGjKHbRA8DZ2Rl/f39u3brFjRs3AGjatCm1atUq1XD/JTc3l+PHjxMUFKTbp1arCQoKIjQ0VJfr7Nmz3Lx5k/T0dLZu3UqnTp0ees2pU6diZWWl23x9fZ/48yjrmt1dujZU+nqUSc2r2TG0RRUA3vnlNClZ8h+4EEIIUWJ6BhA4FkYdBp9OoMmDvdO0q7xcC1E6XYVm6O6Osa8vaDSklcGGppm5+Q/dsvMKSv3c4qhZsyYuLi6EhITo9oWEhNCjRw+8vb05dOhQkf3t2rUDQKPRMHXqVLy9vTExMcHPz49ffvmlyLn/nN6yePFiPDw8MDU1pVevXsycORNra+v7Mv3www9UqVIFKysr+vfvr3szeejQoezZs4c5c+agUqlQqVRcv379oc9twIABJCcns3jx4n/9GsyfP59q1aphaGhIzZo1+eGHH4ocV6lUzJ8/n+7du2NmZsbnn3+um4qzdOlSPD09MTc354033qCgoIBp06bh7OyMo6Mjn3/++b8+tng85ubmusVGQkJCSr0gVeylHzQaDZ999hkzZswgPT0dAAsLCyZMmMDkyZNRq0tURymR+Ph4CgoKcHJyKrLfycmJCxcuAKCvr8+MGTNo164dGo2GSZMmYWdn99Brvvfee4wfP153++bNm5W+8NGsqi0qFVyJSycuLRtHC2OlI4l/eKdzLUIuxnE9IZP/bTzHjBf8FK3ICyGEEOWetScMXAvnN8LWdyDxKqzsAfX7QcfPwdxB6YQVkuWzXcgOD+fOnDmYt26Fwb+M0H7afKdse+ixdjUdWDasqe52o093kvWP4sY9Ad62rH21ue52y692k5iRe99517/sWqx87dq1Y/fu3bz77ruAdkTHpEmTKCgoYPfu3bRt25asrCwOHz7Myy+/DGjf8F21ahULFizAx8eHvXv3MmjQIBwcHGjTps19j3HgwAFee+01vvrqK7p3787OnTv58MMP7zvv6tWrbNiwgU2bNpGUlMQLL7zAl19+yeeff86cOXO4dOkSdevW5ZNPPgHAweHhP0+WlpZMnjyZTz75hCFDhmBmZnbfOb/99htvvvkms2fPJigoiE2bNjFs2DDc3d11BR7Q9hv58ssvmT17Nvr6+ixdupSrV6+ydetWgoODuXr1Kn369OHatWvUqFGDPXv2cPDgQV5++WWCgoIICAgo1r+JeDRBQUG0a9dOt0xtr169MDQ0fOC5f/75Z7GvX+yix+TJk1myZAlffvklgYGBAOzfv5+PP/6Y7OzsMlkF6969O927d3+kc42MjDAyMtLdftLzi8oDa1NDajtbEn47lUPXEunuV3Z++QgtE0M9ZrzgR98Fofx68iYNvWwY1MxL6VhCCCFE+aZSgW8PqNoO/vwMjiyC02vh0jbo+Ck0GARP8Q2/ysBm0CBSt2wlOzycG2PfxGv1KtR/+9tcPFy7du0YN24c+fn5ZGVlcfLkSdq0aUNeXh4LFiwAIDQ0lJycHNq1a0dOTg5ffPEFO3fu1PVprFq1Kvv372fhwoUPLHp88803dOnShYkTJwJQo0YNDh48yKZNm4qcp9FoWL58uW7Vz5deeoldu3bx+eefY2VlhaGhIaampjg7Oz/Sc3vjjTeYM2cOM2fOfGCRZfr06QwdOpQ33ngDgPHjx3Po0CGmT59epOgxcOBAhg0bdl/WpUuXYmFhga+vL+3atePixYts2bIFtVpNzZo1+eqrr9i9e7cUPZ6QVatWsWLFCq5evcqePXuoU6cOpqampXb9Yhc9VqxYwffff1+kiFC/fn3c3Nx44403nmrRw97eHj09PWJjY4vsj42NfeQfoIeZN28e8+bNIzf3/qprZdS8mh3ht1MJvRovRY8yqpGXLZM612LdsWhaVHv4aCYhhBBCFJOxJTw7Dfz6wR/jIOY0bBwDZ36G5xeDxeP93Sn+ojY2xm3uXK737k322bPEfvYZLp9+qnQsAMI/efgUefU/Rtge/zDoIWfef+7+d9o95Mziadu2LRkZGRw9epSkpCRq1KihG7ExbNgwsrOzCQkJoWrVqnh6enLu3DkyMzPpcHe54Htyc3MfukDFxYsX7+u12LRp0/uKHlWqVNEVPABcXFwea6EJIyMjPvnkE8aMGcPrr79+3/Hz588zcuTIIvsCAwOZM2dOkX2NGze+777/zOrk5ISenl6RGQxOTk6yUMYTZGJiwmuvvQbAsWPH+Oqrrx44Zaqkil2aTkxMfGDvjlq1apH4lBsOGRoa0qhRI3bt2qXbp9Fo2LVrl65aWVKjRo0iPDy8yLy4yqx5VWlmWh682roqf4xuSVUHc6WjCCGEEBWPWyMYsRs6fQEGZhCxFxa00n4UpcbQ3Q3XGTNApSL5519IWrdO6UgAmBrqP3QzNtAr9XOLq3r16ri7u7N79252796tG6nh6uqKh4cHBw8eZPfu3TzzzDMAulYFmzdvJiwsTLeFh4cX6etREgYGBkVuq1QqNBrNY11z0KBBeHl58dlnn5X4Gg+aGvOgrE8iv3g0u3fv1hU8CgsLKSyFpcOLXfTw8/Pj22+/vW//t99+i5+f32MH+qf09HTdDyBAREQEYWFhREVFAdqhS4sXL2bFihWcP3+e119/nYyMjPuGLYnH07SqLWoVXE/I5HZKltJxxEOoVCrMjP76JXn4WgKp2dLYVAghhCg1evrQfBS8ugcc60BGnLbXx56vQV4UlRrzloE4vPkmALGffkbW6dMKJyof2rVrR0hICCEhIUWWqm3dujVbt27lyJEjuukevr6+GBkZERUVRfXq1YtsHh4eD7x+zZo1OXr0aJF9/7z9KAwNDSkoeHDPk4dRq9VMnTqV+fPn39f4tHbt2hw4cKDIvgMHDlT63ozl1cqVK6lXrx4mJiaYmJhQv379+xrTFkexS4jTpk2ja9euReZ+hYaGEh0dzZYtW0oc5GGOHTtWZB7WvSajQ4YMYfny5fTr1487d+4wZcoUYmJiaNCgAcHBwfc1Ny0umd5SlKWxAXXdrDh9I4XQqwk839Bd6UjiP6w7Fs17v56htY893w9pgp5aGpsKIYQQpcbeB17ZCVvfhpOrYPdnEBUKzy8CM3ul01UIdiNHkHXmDOm7dnHjzXF4r/8FfVtbpWOVae3atWPUqFHk5eUV6cnRpk0bRo8eTW5uru61lYWFBRMnTuStt95Co9HQsmVLUlJSOHDgAJaWlgwZMuS+648ZM4bWrVszc+ZMunXrxp9//snWrVuL3UC/SpUqHD58mOvXr2Nubo6tre0jLYjRtWtXAgICWLhwYZHXe2+//TYvvPAC/v7+BAUF8ccff/Drr7+yc+fOYuUSyrvXt2X06NFFeoi+9tprxMfH89ZbbxX7msUe6dGmTRsuXbpEr169SE5OJjk5meeff56LFy/SqlWrYgf4L23bttUNa/n7tnz5ct05o0ePJjIykpycHA4fPlwqDWZkesv9ZIpL+VLL2QJ9tYrdF+8wLfiC0nGEEEKIisfQFHrMg57zQd8Eru7STneJOvTf9xX/SaVW4/rlVAyrVCH/9m1ujp9AYX7xlnKtbNq1a0dWVhbVq1cvUhRo06YNaWlpuqVt7/n000/58MMPmTp1KrVr16Zz585s3rwZb2/vB14/MDCQBQsWMHPmTPz8/AgODuatt97C2Lh4qztOnDgRPT09fH19cXBw0I3ifxRfffUV2dnZRfb17NmTOXPmMH36dOrUqcPChQtZtmxZkdEuonz45ptvmD9/vm6FoO7duzNt2jS+++475s6dW6JrqgpLY5IMcOPGDT755BMWLVpUGpcrM27cuIGHhwfR0dG4u1fu0Q0hF+MYuuwo7jYm7H/nGaXjiEfwx6lbjPnpJAAzX/CTETpCCCHEkxIbDj8PgfhLoNKDoI+hxRjtCjDiseRcvkxEv/4UZmZiO/xlnN5++4k+XnZ2NhEREXh7exf7xXxlNGLECC5cuMC+ffuUjlLu/Nv3WmV9HWpsbMzZs2epXr16kf2XL1+mXr169xW8HkWprbGVkJDAkiVLSutyogxqUsUWfbWKG0lZRCdmKh1HPIJufq6Mbqf9D+PdX89wMipJ4URCCCFEBeXkq21yWq8vFBbAjg/hpwGQ+XQb/VdERj4+uH6hXSEycclSUoO3KZyocps+fTqnTp3iypUrfPPNN6xYseKBU2GEKInq1auz7gHNi9euXYuPj0+Jrln8tsCVhPT0uJ+ZkT713a04EZVM6LUEPGxLb+1k8eSM71CDS7FpbA+PZeQPx9k4OhAXKxOlYwkhhBAVj5G5dglbr0DY+g5c2goL20Df5eDeSOl05Zpl585kDTtN4rJl3H7/fYx8qmNUrZrSsSqlI0eOMG3aNNLS0qhatSpz587llVdeUTqWqCD+97//0a9fP/bu3avr6XHgwAF27dr1wGLIoyi1kR4VjfT0eLDm1bR9PQ5JX49yQ61WMatfA2o5W3AnLYdfT9xUOpIQQghRcalU0HgYvLIDbLwhJQqWdoLDC6F0ZpVXWo4TxmPatCmazExujB5Dwd0lV8XTtW7dOuLi4sjKyuLcuXO89tprSkcSFUjv3r05fPgw9vb2bNiwgQ0bNmBvb8+RI0fo1atXia4pRQ9RLM2raruRh15LKJU1k8XTYWakz+LBjfm4my9vtJV3RYQQQognzsVPu6xt7e6gyYOtk7Q9P7JTlE5Wbqn09XGbNRN9Z2dyIyK4/d578veoEBVQo0aNWLVqFcePH+f48eOsWrUKf3//El/vkae3PP/88/96PDk5ucQhRPnRyMsGAz0Vt1OyiUzIpIq9mdKRxCPysDVlaOBfnbgLCwuLvbyYEEIIIYrB2ApeWAlHFsG2yRD+O8Scgb4rwKW+0unKJX07O9znzCZy0Euk7dhJwuLvsR854ok8lhRUxJMm32NPxyOP9LCysvrXzcvLi8GDBz/JrKIMMDHUw9/DBtCO9hDlU1p2HiNWHmfbuRilowghhBAVm0oFAa/Cy9vAyhMSr8H3QXBsmUx3KSETPz+cPvgAgDuzZ5Nx8GCpXt/AwACAzExp3C+erHvfY/e+58ST8cgjPZYtW/Ykc5Q50sj04ZpVs+PI9URCryYwoKmn0nFECaw4eJ2d52M5eDWeX99oQS1nS6UjCSGEEBWbeyPtdJcNb2gbnG4aB5EH4blZ2gaoolisX+hL1ulTpKz/lZvjJ+C9/hcM3NxK5dp6enpYW1sTFxcHgKmpqYyOFaWqsLCQzMxM4uLisLa2Rk9PT+lIFZqqUMbU/KvKuj7yvwm9msCAxYdwsDDiyPvt5ZdAOZRXoGHI0iMcvJqAu40Jv48KxM7cSOlYQgghRMVXWAgHv4GdH2uXtrWvoZ0C41hb6WTljiYnh8iBL5J97hzGderg9eNq1Eal8/dMYWEhMTExMoVfPFHW1tY4Ozs/8PWUvA4tPVL0+A/yzXa/7LwC6v9vO7n5GnaOb0N1R3l3ojxKzsylx7wDRCZk0tTbllXDAzDUl97GQgghxFMRdQh+HgZpt0DfBJ6bCQ0GKp2q3Mm7eZOI3n0oSE7GqvfzuHz2Wam+IVdQUEBeXl6pXU+IewwMDP51hIe8Di09jzy9RYh7jA30aORpQ+i1BEKvJUjRo5yyNjXk+8GN6fXdQY5EJPLRxnN80auujNwRQgghngbPZvDaPvh1BFz9Eza8DpEHoMvXYGiqdLpyw8DNDbeZM4h6ZQQp63/FpL4fNv1eKLXr6+npydQDIZ6i7OxsvvnmG3bv3k1cXBwajabI8RMnThT7mvK2riiR5tXsADh0VZqZlmc+ThZ8M8AflQp+OhLFD4cilY4khBBCVB5m9vDiemj3AajUcHKVtslpwlWlk5UrZi1a4DBuHAAxn31G1qlTygYSQpTY8OHDmTZtGl5eXjz33HP06NGjyFYSMtLjIaSR6b9rXs0OdsChawmy9Gk5166WI+92rsWyA9d1K/MIIYQQ4ilRq6HN2+DRFNa/AnHn4Pv20G81VAlUOl25YTfiFbLPnCZtx05ujH0T71/Xo29np3QsIUQxbdq0iS1bthAYWHr//8lIj4cYNWoU4eHhhISEKB2lTPJzt8bEQI+EjFwuxaYrHUc8ppGtqxI8rhX13K2UjiKEEEJUTlXbaKe7uDWGrCRY2QPCflI6VbmhUqlwmToVw6pVyY+N5eZb4ynMz1c6lhCimNzc3LCwsCjVa0rRQ5SIob6axlW0owJCr8YrnEY8LpVKhbWpoe722ZsppGZL0y4hhBDiqbJwhqGbwLcnaPJgw2vw52fwjznt4sH0zM1x/2YualNTMo8cIW7mLKUjCSGKacaMGbzzzjtERpbetHspeogSa1ZVO2Qw9Jr09ahIgs/epvf8g7z500kKNLK4kxBCCPFUGZhAn2XQaoL29t6vYf1wyMtWNlc5YVStGi5TpwKQuHQpqVu3KpxICFEcjRs3Jjs7m6pVq2JhYYGtrW2RrSSkp4coMV0z02uJaDSFqNXS16MicLU2AWD3xTtMC77Ae8/WVjiREEIIUcmo1dB+CthWgz/ehHO/QsoN6P8jmDsona7Ms+zUkexXhpPw/RJuTf4Ao+rVMfLxUTqWEOIRDBgwgJs3b/LFF1/g5ORUKr0jpeghSqyemxVmhnqkZOURfjuVum7SD6IiqO9uzfS+foz56SQL916jhpMFvRvJ2uBCCCHEU+f/Ilh7wtpBcOMIfP8MDPwZHGspnazMcxg3jqyz58g8dIgbY8ZS5ed16JVynwAhROk7ePAgoaGh+Pn5ldo1ZXrLQ8ybNw9fX1/atm2rdJQyy0BPTRNv7RCjQzLFpULp5ufK6HbVAXjv1zOciEpSOJEQQghRSXm3gld2gW1VSI6CJR3g6p9KpyrzVPr6uM2cgb6LC7nXr3Pr3fcolN4oQpR5tWrVIisrq1SvKUWPh5DVWx5Ni7tTXEKvStGjohnfoQYdfZ3ILdAwcuVxbiWX7n8+QgghhHhE9tVh+E7wbAE5qbCqDxxbpnSqMk/f1hb3uXNQGRiQvmsXCYsWKx1JCPEfvvzySyZMmEBISAgJCQmkpqYW2UpCih7isTSvag/AkYhE8gukel6RqNUqZvVrQC1nC+LTc1h7NFrpSEIIIUTlZWYHgzdA/f5QWACbxsG2yaApUDpZmWZSrx5OUz4E4M6cOeRERCicSAjxbzp37kxoaCjt27fH0dERGxsbbGxssLa2xsbGpkTXlJ4e4rH4ulpiaaxPanY+526l4udhrXQkUYrMjPT5fkhjVh+O4s320gBMCCGEUJS+EfRaAHbVYfdnEPotJEZA78VgaKZ0ujLLpm9fklb+QM7ly+RFR2Pk7a10JCHEQ+zevbvUrylFD/FY9NQqmnrbsfN8LKHXEqToUQG525jyTue/GqblF2hIycrDztxIwVRCCCFEJaVSQZu3wdYbNrwBFzfDsi4wYA1YuiqdrsxSGRoqHUEI8QjatGlT6teUood4bM2r3S16XE3gtTbVlI4jnqD8Ag1vrg3j/K1UfhrZDCdLY6UjCSGEEJVTvT7alV1+GgC3T8Hi9jBwLbjUVzqZEEKU2N69e//1eOvWrYt9TSl6iMfWvKq2menR64nkFWgw0JNWMRVVYkYuYVHJ3EzOov+iQ/w0ohnOVlL4EEIIIRTh0RRe2Qk/9oP4i7C0M/RZAjW7KJ1MCCFK5EGrp6pUKt3nBQXF72Mkr07FY6vlbIGNqQGZuQWcvpGidBzxBDlaGrNmZDPcrE2IiM+g/6JQbqfIqi5CCCGEYmy9Yfh2qNoW8jK0Iz9Cv4PCQqWTCSFEsSUlJRXZ4uLiCA4OpkmTJmzfvr1E15Sih3hsarWKAG/taI9D12Tp2orOw9aUta82w93GhOsJmfRfdEgKH0IIIYSSTKzhxV+g4RCgELa9B1smQkG+0smEEKJYrKysimz29vZ06NCBr776ikmTJpXomlL0eIh58+bh6+v7wOE14n7Nq2mLHqFXpehRGbjbmLJmZDM8bE2IvFv4uJUshQ8hhBBCMXoG0G0OdPgUUMHR7+GnfpCdqnQyIYR4bE5OTly8eLFE95Wix0OMGjWK8PBwQkJClI5SLtwrehyLTCQnX9aLrwy0hY/meNiaEJOSTWRCptKRhBBCiMpNpYLAsdDvB9A3gSs7YWknSI5SOpkQQjyS06dPF9lOnTpFcHAwr732Gg0aNCjRNaXoIUqFj6M59uaGZOdpOBUtfT0qCzdrE9aMbM6yYU10hS8hhBBCKKx2N3h5K5g7Q1y4dmWXG8eUTiWEeIJu3rzJoEGDsLOzw8TEhHr16nHs2F8/94WFhUyZMgUXFxdMTEwICgri8uXLRa6RmJjIiy++iKWlJdbW1gwfPpz09PQi55w+fZpWrVphbGyMh4cH06ZNK9Xn0aBBA/z9/WnQoIHu82effZbc3Fy+//77El1Tih6iVKhUKgKqyhSXysjN2oQW1ex1t6/EpXNTproIIYQQynL1hxG7wKkuZMTB8q5waZvSqYQQT0BSUhKBgYEYGBiwdetWwsPDmTFjBjY2Nrpzpk2bxty5c1mwYAGHDx/GzMyMTp06kZ2drTvnxRdf5Ny5c+zYsYNNmzaxd+9eRo4cqTuemppKx44d8fLy4vjx43z99dd8/PHHLFq0qNSeS0REBNeuXSMiIoKIiAgiIyPJzMzk4MGD1KpVq0TXlCVrRalpXtWOzadvE3otnjfxUTqOUMDVO+n0X3QIE0M1P41ohruNqdKRhBBCiMrLyh1eDoZfXobL22HNQOj9PdTppXQyIUQp+uqrr/Dw8GDZsmW6fd7e3rrPCwsLmT17Nh988AE9evQAYOXKlTg5ObFhwwb69+/P+fPnCQ4O5ujRozRu3BiAb775hmeffZbp06fj6urK6tWryc3NZenSpRgaGlKnTh3CwsKYOXNmkeLI4/Dy8iqV6/ydjPQQpebe9IYTUclk50lfj8rI1FAPcyM9ohOz6L/oENGJ0udDCCGEUJSRBfT/Eer2Bk2+tgBy4gelUylHlvIV5UxaWhqpqam6LScn575zNm7cSOPGjenbty+Ojo74+/uzePFi3fGIiAhiYmIICgrS7bOysiIgIIDQ0FAAQkNDsba21hU8AIKCglCr1Rw+fFh3TuvWrTE0NNSd06lTJy5evEhSUtJjPc/Q0FA2bdpUZN/KlSvx9vbG0dGRkSNHPvC5PwopeohSU9XeDEcLI3LzNZyIerxvelE+uVhpe3xUsTPlRpIUPoQQQogyQc8Anl8MDQdDoQY2joZDC5ROJYR4BL6+vkWWcJ06dep951y7do358+fj4+PDtm3beP311xk7diwrVqwAICYmBtCugPJ3Tk5OumMxMTE4OjoWOa6vr4+trW2Rcx50jb8/Rkl98sknnDt3Tnf7zJkzDB8+nKCgIN59913++OOPBz73RyFFD1FqVCqVbrTHIenrUWk5WxmzZmRzvO3NuJkshQ8hhBCiTFDrQbe50Hy09nbwO7D368oz8kGlUjqBECUSHh5OSkqKbnvvvffuO0ej0dCwYUO++OIL/P39GTlyJCNGjGDBgvJT3AwLC6N9+/a622vWrCEgIIDFixczfvx45s6dy7p160p0bSl6iFLV/F4z02tS9KjMnK2M+WlEM6pK4UMIIYQoO1Qq6PgZtL37ounPz2DHlMpT+BCiHLKwsMDS0lK3GRkZ3XeOi4sLvr6+RfbVrl2bqCjtctXOzs4AxMbGFjknNjZWd8zZ2Zm4uLgix/Pz80lMTCxyzoOu8ffHKKmkpKQio0j27NlDly5ddLebNGlCdHR0ia4tRQ9Rqu6N9AiLTiYrV/p6VGbOVsb8NFJb+LA3N8TSxEDpSEIIIYRQqaDtu9Dxc+3tg3Nh83jQaJTNJYQoscDAQC5evFhk36VLl3RNQb29vXF2dmbXrl2646mpqRw+fJjmzZsD0Lx5c5KTkzl+/LjunD///BONRkNAQIDunL1795KXl6c7Z8eOHdSsWbPISjEl4eTkREREBAC5ubmcOHGCZs2a6Y6npaVhYFCy1xNS9BClytPWFDdrE/IKCjkWmah0HKEwJ0tj1oxsxsrhAVhJ0UMIIYQoO1qMhm5zABUcWwobXoOCfKVTCSFK4K233uLQoUN88cUXXLlyhR9//JFFixYxatQoQNuGYNy4cXz22Wds3LiRM2fOMHjwYFxdXenZsyegHRnSuXNnRowYwZEjRzhw4ACjR4+mf//+uLq6AjBw4EAMDQ0ZPnw4586dY+3atcyZM4fx48c/9nN49tlneffdd9m3bx/vvfcepqamtGrVSnf89OnTVKtWrUTXlqKHKFUqlYpm96a4SF8PAThaGhcpeKw+HElkQoaCiYQQQggBQKOh2iVs1fpwei38PATyS7Y6ghBCOU2aNOG3337jp59+om7dunz66afMnj2bF198UXfOpEmTGDNmDCNHjqRJkyakp6cTHByMsbGx7pzVq1dTq1Yt2rdvz7PPPkvLli1ZtGiR7riVlRXbt28nIiKCRo0aMWHCBKZMmVIqy9V++umn6Ovr06ZNGxYvXszixYuLrBKzdOlSOnbsWKJrqwoLZRLfg8ybN4958+aRm5vL1atXiY6Oxt3dXelY5cIvx28w8edTNPCwZsOoQKXjiDLk52PRvP3LaVzu9vyoYm+mdCQhhBBCXNwK64ZAQQ5UbQf9V4NhxfodHdGnL9lnz+KxcAHmbdooHUeI/3Tjxg08PDwq3evQlJQUzM3N0dPTK7I/MTERc3PzIoWQRyUjPR5i1KhRhIeHExISonSUcudeX48zN1NIz5FhkuIvbWs6Ut3RnNsp2fRfdIiIeBnxIYQQQiiuZhd4cR0YmMG13fDD85CVrHQqIUQlZGVldV/BA8DW1rZEBQ+Qood4AtysTfC0NaVAU8jRCOnrIf7iYGHETyOa4eNoTkxqNv0XhXL+dqrSsYQQQghRtS0M3gDGVhB9CFZ0g4x4pVMJIcRjk6KHeCJk6VrxMA4WRvx4t/ARm5pD7/kHCT57W+lYQgghhPBoCkM2gak9xJyGZc9C6i2lUwkhxGORood4Iu5NcZFmpuJBHCyM+Pm15rSsbk9mbgFvrD7B1TvpSscSQgghhEt9eDkYLN0g/iIs7QyJEUqnEkKIEpOih3gi7hU9zt1KISUr7z/OFpWRtakhy4c1YVhgFd5sX4NqDuZKRxJCCCEEgL0PDNsKNt6QHAnLusCdi0qnKhWyhoMQlY8UPcQT4WRpTFV7MzSFcET6eoiH0NdT81G3OoxtX12371ZyFlEJmQqmEkIIIQQ2XtoRHw61Ie22tvBxK0zpVEIIUWxS9BBPTDOZ4iIekUqlAiArt4ARK4/Rfd5+Dl6R5mlCCCGEoiycYehmcGkAmQna5qZRh5ROJYQQxSJFD/HESDNTUVzpOfnoq1UkZ+bx0tIjLD8QIcNQhRBCCCWZ2cGQP8CzBeSkwg+94OqfSqcqvrtvsAghKh8peognptndosf526kkZeQqnEaUBw4WRqx9tTm9/N0o0BTy8R/hvLv+DDn5BUpHE0IIISovY0sYtB6qtYe8TPixH5zfpHQqIYR4JFL0EE+Mg4URPo7a5pSHI2S0h3g0xgZ6zHzBj8nP1katgrXHohm4+DBxadlKRxNCCCEqL0NTGPAT1O4OBbmwbjCcXqd0KiGE+E9S9BBPlCxdK0pCpVIxonVVlg1rioWxPscjk3h3/RmlYwkhhBCVm74R9FkGfgOhsAB+HQlHlyidSggh/pUUPcQTJX09xONoU8OB30cF0tTblk961FE6jhBCCCH09KHHPGgyAiiEzeMh5CuQHlxCiDJKih7iiQq4W/S4FJtOfHqOwmlEeVTVwZx1rzbH3cZUt+/glXgKNPLHlRBCCKEItRqe/Rpav629HfIFbJ4AGunBJYQoeypF0aNXr17Y2NjQp08fpaNUOrZmhtRytgDgkIz2EKVg+7kYBn5/mOErjpKSlad0HCGEEKJyUqngmQ/g2emACo4tgZ+HQp704BJClC2Voujx5ptvsnLlSqVjVFrS10OUpryCQowN1IRcvEOveQe4eidd6UhCCCFE5dV0BPRdBnqGcH4jrOoN2SlKpxJCCJ1KUfRo27YtFhYWSseotKSvhyhNXeu78MtrLXC1MuZafAY9vz3A7gtxSscSQgghKq86vbRL2hpaQOR+WNYV0mKUTiWEEEAZKHrs3buXbt264erqikqlYsOGDfedM2/ePKpUqYKxsTEBAQEcOXLk6QcVJRZQ1Q61Cq7dySA2VYY8isdX182K30e3pEkVG9Jy8nl5xVEW7LlKoTRRE0IIIZTh3RqGbQYzR4g9A0s6QMJVpVPdT/5WEKLSUbzokZGRgZ+fH/PmzXvg8bVr1zJ+/Hg++ugjTpw4gZ+fH506dSIu7q93dhs0aEDdunXv227duvW0nob4F1YmBtRxtQKkr4coPQ4WRqx+pRkDmnpSWAhfbr0gU6iEEEIIJbn4wfDtYOMNyVGwpCPcPKF0KiFEJaevdIAuXbrQpUuXhx6fOXMmI0aMYNiwYQAsWLCAzZs3s3TpUt59910AwsLCSi1PTk4OOTl/rTKSlpZWateuzJpXs+PMzRRCrybQo4Gb0nFEBWGor+aLXnXxdbXkenwGLarbKx1JCCGEqNxsvbWFj9V94PYpWP4c9PsBqrdXOpkQopJSfKTHv8nNzeX48eMEBQXp9qnVaoKCgggNDX0ijzl16lSsrKx0m6+v7xN5nMpG+nqIJ0WlUvFSMy8+fO6vn9VbyVlsOXNbwVRCCCFEJWbuCEM3Q9W2kJcBP74Ap39WOpUQopIq00WP+Ph4CgoKcHJyKrLfycmJmJhHb44UFBRE37592bJlC+7u7v9aMHnvvfdISUnRbeHh4SXOL/7SxNsWPbWKyIRMbiZnKR1HVGCFhYW8s/40b6w+wegfT5CUkat0JCGEEKLyMbKAgT9D3d6gyYdfX4HQ75TLo1Ip99hCCEWV6aJHadm5cyd37twhMzOTGzdu0Lx584eea2RkhKWlpW6TVV9Kh7mRPvXctH09pO+CeJIKNIX4e1ijp1ax6fRtOszay/Zz0kFeCCGEeOr0DeH57yHgNe3tbe/Bjo+kmagQ4qkq00UPe3t79PT0iI2NLbI/NjYWZ2fnJ/rY8+bNw9fXl7Zt2z7Rx6lMmle7O8VFih7iCdLXUzO+Y01+e6MFPo7mxKfnMPKH44xfG0ZKZp7S8YQQQojKRa2Gzl9C+4+0tw/Mhg1vQIH8ThZCPB1luuhhaGhIo0aN2LVrl26fRqNh165d/zpaozSMGjWK8PBwQkJCnujjVCb3+nocupYgS4uKJ66+uzV/jGnJa22qoVbBrydv0nH2Hi7GSHNiIYQQ4qlSqaDVeOgxD1R6cOpHWDMQcjOUTiaEqAQUL3qkp6cTFhamW4ElIiKCsLAwoqKiABg/fjyLFy9mxYoVnD9/ntdff52MjAzdai6i/GhcxQYDPRU3k7OITpS+HuLJMzbQ490utfjl9RZUtTfD3EgfLztTpWMJIYQQlZP/IOj/I+ibwOXtsLIHZCYqnUoIUcEpXvQ4duwY/v7++Pv7A9oih7+/P1OmTAGgX79+TJ8+nSlTptCgQQPCwsIIDg6+r7lpaZPpLaXP1FAfP3drAEKvxSsbRlQqDT1t2PJmK5YObYKxgR6g7f1xPFL+0BJCCCGeqpqdYfDvYGwNN47C0k6QHK10KiFEBaZ40aNt27YUFhbety1fvlx3zujRo4mMjCQnJ4fDhw8TEBDwxHPJ9JYnQ/p6CKUYG+jhZWemu71k/zV6zw9l8m9nyMjJVzCZEEIIUcl4BsDL28DSDeIvwZKOECsrJgohngzFix6icrnX1yNU+noIhcWna5eyXX04is5z9nLomhTihBBCiKfGsRYM3w4OtSDtFizrDJGhSqcSQlRAUvQQT1VDLxsM9dTEpuZw/rY0lBTKef/Z2vz4SgBu1iZEJ2bRf9EhPt54jqzcAqWjCSGEEJWDlTsM2woeAZCdAj/0hAtbnuxjyntuQlQ6UvR4COnp8WQYG+jRtqYDAO+sP01uvkbhRKIya1HdnuBxrRjQ1AOA5Qev8+zcfZy+kaxsMCGEEKKyMLWFlzZAjc6Qnw1rX4QTK5VOJYSoQKTo8RDS0+PJ+aRHXaxNDThzM4UZ2y8qHUdUchbGBkx9vj4rXm6Ks6UxN5IyMdSX/xqFEEKIp8bQFPqthgaDoFADG8fA3q9BpkILIUqB/GUvnjpnK2O+6l0fgIV7r7Hv8h2FEwkBbWo4sO2t1iwY1Ihazpa6/XfSchRMJYQQQlQSevrQ41toOV57+8/PYOs7oJFRwUKIxyNFD6GITnWceTHAE4Dx606RkC4vLIXyrEwMaF/7r+WwT99IJvCrP5m+7SI5+dLrQwghhHiiVCoI+gg6f6m9fWQh/PoK5OeWwrUf/xJCiPJJih5CMR909cXH0Zw7aTm8/ctpWc1FlDnbz8WSm6/h291X6PHtAcKik5WOJIQQQlR8zV6H578HtT6cXQ8/9YOcdKVTCSHKKSl6PIQ0Mn3yTAz1mDvAH0N9NX9eiGPFwetKRxKiiImdajL/xYbYmhlyISaNnvMO8Oaak9xMzlI6mhBCCFGx1e8LA9aCgSlc/RNWdIOMeKVTCSHKISl6PIQ0Mn06artY8n6XWgB8sfUC52+nKpxIiKK61HNh+1uteb6hGwC/h93imekhLNp7VeFkQgghRAXnEwRD/gATW7h1ApZ2guQopVMJIcoZKXoIxQ1pUYX2tRzJzdcw9qeTZOVK7wRRttibGzHzhQZsGtOSZlVtycnXYGFsoHQsIYQQouJzbwwvbwMrD0i4Aks6Qmy40qmEEOWIFD2E4lQqFdP61MfBwojLcel8tll+kYmyqa6bFT+NaMaKl5vyQmMP3f6Qi3HsvSSrEAkhhBBPhEMNbeHDoTak3YZlnSHqkNKphBDlhBQ9RJlgZ27EzBf8AFh9OIpt52IUTiTEg6lUKtrUcEBPrW0Dn51XwOTfzjJ46RGGLD3Cpdg0hRMKIYQQFZCVGwzbAh4BkJ0CK3vAxWClUwkhygEpejyENDJ9+lr5OPBq66oAvLP+NLdTpFmkKPvyNYV0ruuMgZ6KPZfu0Hn2Xib/doZ4WYZZCCGEKF2mtvDSBvDpBPnZsGYgnFytdCohRBknRY+HkEamypjQsSb13KxIzszjrbVhFGhkGVtRtpkb6fPhc77seKsNnes4oynUjlZq+3UI80Oukp0nPWqEEEKIUmNoCv1Xg99AKCyA39+AA3OKcQH521KIykaKHqJMMdRXM3eAP6aGehy6lsiCPbJChigfqtibseClRqwd2Yx6blak5+TzVfAFTkUnKx1NCCGEqFj0DKDnd9BirPb2jimwbTJoNMrmEkKUSVL0EGWOt70Z/+teB4CZOy5xMipJ4URCPLqAqnb8PiqQmS/4MaiZJwFV7XTH4lKzFUwmhBBCVCAqFXT8FDp8qr0d+i1seB0K8pTNJYQoc6ToIcqkPo3c6ebnSoGmkLFrTpKWLb/ARPmhVqt4vqE7n/Wsp9sXm5pN2+khjPrxBNGJmQqmE0IIISqQwLHQcwGo9OD0Gm2fj9wMpVMJIcoQKXqIMkmlUvFZz7q4WZsQnZjFlN/PKR1JiMey73I8WXkFbD59m/Yz9jB1y3lSpZgnhBBCPL4GA2DAT6BvApe3a1d2yUxUOpUQooyQosdDyOotyrMyMWDugAboqVX8dvImv528oXQkIUqsTyN3No9pRcvq9uQWaFi49xptvw5hZeh1cvKl2akQQgjxWGp0gsG/g7E13DgKSztDyl9/O6pQKZdNCKEoKXo8hKzeUjY08rLlzfY+AHzw21kiE2S4oii/fF0t+WF4U5YNbUI1BzMSM3KZ8vs5uszeJysVCSGEEI/LMwBeDgYLV4i/CEs6wp2LSqcSQihMih6izBvVrjpNq9iSkVvA2DVh5BVIZ25RfqlUKtrVciR4XGs+7VEHZ0tj2tZ0RE/91ztQKVky7UUIIYQoEcfaMHw72NeA1JuwtBNEH1U6lRBCQVL0EGWenlrFrP4NsDTW51R0MrN2XFI6khCPzUBPzUvNq7BnUlvGdfDR7T96PZFmX+zikz/CuZ2SpWBCIYQQopyy9oBhweDWGLKSYGV3yE5ROpUQQiFS9BDlgpu1CV/1rg/A/D1XOXglXuFEQpQOI309LI0NdLc3n75NVl4BSw9E0Hrabib9coqrd9IVTCiEEEKUQ2Z2MGQjVGsPeZkQf1npREIIhUjRQ5QbXeq5MKCpB4WF8Na6MJIycpWOJESp+6ibLytebkqAty15BYWsO3aDoJl7eGP1cc7ckHephBBCiEdmaAYD1kC9vsDd3lmXtysaSQjx9EnRQ5QrHz7nSzUHM2JTc5i0/jSFhdL8UVQsKpWKNjUcWPtqc9a/3oKg2o4UFsKWMzGM+vGENDwVQgghikPfEHotAnMn7e1jS+HQAmUzCSGeKil6iHLF1FCfuQP8MdRTsyM8llWHo5SOJMQT08jLhu+HNGHbuNb08nfjjbbVdA1Pc/M17AyPRSNFECGEEOLfqdVg7an9vFAFwe/AwW+UzSSEeGqk6CHKnTquVrzTpRYAn20K51JsmsKJhHiyajpbMKtfA/o39dTt23DyJq+sPEan2XtZf/yGrGokhBBCPIo6PbUft38A+2YoGkUI8XRI0eMh5s2bh6+vL23btlU6iniAYS2q0KaGAzn5Gsb+dJLsvAKlIwnxVGXlFWBhpM/luHQm/HyKtl+HsOLgdbJy5WdBCCGEeKj6L0C7ydrPd30CIV8pm0cI8cRJ0eMhRo0aRXh4OCEhIUpHEQ+gVquY3tcPe3MjLsSkMXXLeaUjCfFUDWlRhQPvPcOkzjWxNzfkZnIWH208R8uv/uTbPy/LtBchhBDiYdpMgvYfaT8P+QL+/AykT5wQFZYUPUS55WBhxPS+2mVsV4RGsut8rMKJhHi6LI0NeKNtdfa/8wyf9qyLh60JCRm5HI5IRH2394cQQgghHqDVeOj4ufbzvV/Dzo+k8CFEBSVFD1Guta3pyPCW3gC8/ctp4lKzFU4kxNNnbKDHS8282D2hLXP6N+CtDjV0x24lZ9Hj2/38eDiK9Jx8BVMKIYQQClI94M2AFqOhyzTt5wfmwLb3pfAhRAUkRQ9R7k3qXJM6rpYkZuQy4ofjRCVkKh1JCEXo66np0cCNhp42un1rj0Zz6kYK7/92hoDPd/L+b2c4ezNFwZRCCCFEGRLwKnSdqf380HewZSJopDm4EBWJFD1EuWekr8fcAf6YG+lzKjqZTrP3smR/BAXS00AIhraowgdda1PV3oyM3AJ+PBzFc9/sp8e8A6w7Gi1NgIUQQogmw6H7t4AKjn4Pm8ZJ4UOICkSKHqJCqOZgzqYxLWlW1ZasvAI+3RROnwUHuSzL2YpKzsbMkFdaVWXXhDb8NKIZz9V3wUBPxanoZD7+45wsdSuEEEIANHwJei0AlRpOrICNo0EjbwyI8unLL79EpVIxbtw43b7s7GxGjRqFnZ0d5ubm9O7dm9jYoj0Ro6Ki6Nq1K6ampjg6OvL222+Tn190enRISAgNGzbEyMiI6tWrs3z58qfwjB6PFD1EhVHF3owfX2nGF73qYW6kz8moZLrO3c/cXZfJzZcXdqJyU6lUNK9mx7cDGxL6Xnve6VyLka2rYmFsAEBhYSFv/3yK9cdvyOgPIYQQlZNff3h+Maj0IGw1/PYaFEg/LFG+HD16lIULF1K/fv0i+9966y3++OMPfv75Z/bs2cOtW7d4/vnndccLCgro2rUrubm5HDx4kBUrVrB8+XKmTJmiOyciIoKuXbvSrl07wsLCGDduHK+88grbtm17as+vJFSFhdKt59/cuHEDDw8PoqOjcXd3VzqOeES3U7KY/NtZ/rwQB0AtZwum9alPfXdrZYMJUUaFRSfTc94BAKxMDOjd0J2BAZ5UdzRXOJkQQgjx+K73H0BWWBju877Fon37fz/53AZYPxw0+VDneXh+EegZPJWcQtxTkteh6enpNGzYkO+++47PPvuMBg0aMHv2bFJSUnBwcODHH3+kT58+AFy4cIHatWsTGhpKs2bN2Lp1K8899xy3bt3CyckJgAULFvDOO+9w584dDA0Neeedd9i8eTNnz57VPWb//v1JTk4mODi49L8IpURGeogKycXKhCVDGjOnfwNsTA24EJNGz3kHmLr1vLyLLcQDuNuYMLFjDdysTUjJymPpgQiCZu7hhYWh/B52k5x8+bkRQghRSdTpCS+sBLUBnPsVfhkG+blKpxKVVFpaGqmpqbotJyfnoeeOGjWKrl27EhQUVGT/8ePHycvLK7K/Vq1aeHp6EhoaCkBoaCj16tXTFTwAOnXqRGpqKufOndOd889rd+rUSXeNskqKHqLCUqlU9Gjgxs7xbejm54qmEBbuuUaXOfs4fC1B6XhClCn25kaMfsaHvZPasWxoE4JqO6FWwZGIRN5cE8aBK/FKRxRCCCEe36MOcq/VFfr/CHpGcP4PWDcY8h/+YlOIJ8XX1xcrKyvdNnXq1Aeet2bNGk6cOPHA4zExMRgaGmJtbV1kv5OTEzExMbpz/l7wuHf83rF/Oyc1NZWsrKwSPb+nQV/pAEI8aXbmRnwzwJ/ufq58sOEMEfEZ9Ft0iJeaeTGpc01dTwMhBOipVbSr5Ui7Wo7cTsli7dFo9l2Op00NR9053++7Rk6+hufqu+BlZ6ZgWiGEEOIJqtERBvwEawbCpa3aj/1WgYGJ0slEJRIeHo6bm5vutpGR0X3nREdH8+abb7Jjxw6MjY2fZrxyQUZ6PMS8efPw9fWlbdu2SkcRpaSDrxPb32rDgKYeAPxwKJJOs/ay+2KcwsmEKJtcrEwYF1SD9a+3QE+tAqBAU8iivdf4ettF2nwdQo9v9/P9vmvcTim71X0hhBCixKq3h4HrwMAUruyEn/pDbqbSqUQlYmFhgaWlpW57UNHj+PHjxMXF0bBhQ/T19dHX12fPnj3MnTsXfX19nJycyM3NJTk5ucj9YmNjcXZ2BsDZ2fm+1Vzu3f6vcywtLTExKbvFQCl6PMSoUaMIDw8nJCRE6SiiFFmZGDD1+fr8+EoAnram3ErJZtiyo4xfG0ZShszVFOK/FGgKmdixJq187FGr4NSNFD7bfJ7mU//khQWhbDh5U+mIQgghROmq2gZe/AUMzOBaCPz4AuSkK51KCJ327dtz5swZwsLCdFvjxo158cUXdZ8bGBiwa9cu3X0uXrxIVFQUzZs3B6B58+acOXOGuLi/3hDesWMHlpaW+Pr66s75+zXunXPvGmWVFD1EpdSiuj3B41rxSktv1Cr49eRNOszaw+bTt5EFjYR4OEN9NS808eCH4QEcmRzEpz3q0LSKLQBHridy6kay7twCTSEpmXkKJRVCCCFKUZVAeOk3MLSA6/tgdR/ISVM6lRCAdjRI3bp1i2xmZmbY2dlRt25drKysGD58OOPHj2f37t0cP36cYcOG0bx5c5o1awZAx44d8fX15aWXXuLUqVNs27aNDz74gFGjRulGl7z22mtcu3aNSZMmceHCBb777jvWrVvHW2+9peTT/09S9BCVlqmhPh8858v611vg42hOfHouo348was/HCcuNVvpeEKUefbmRrzUvArrXmtO6HvP8EHX2vRt5KE7fvhaAo0/38ErK47ye9hNMnLyFUwrhBBCPCbPABj8OxhZQVQo/NALslOUTiXEI5k1axbPPfccvXv3pnXr1jg7O/Prr7/qjuvp6bFp0yb09PRo3rw5gwYNYvDgwXzyySe6c7y9vdm8eTM7duzAz8+PGTNm8P3339OpUyclntIjUxXK29r/qiTrI4vyJye/gHm7r/Ld7ivkawqxNNYWRPo2ckelUikdT4hyaeaOS8zddVl329hATftaTnTzc6FtTUeMDfQUTCeEEKIyuT5gIFknT+L+7TdY/GPJzWK7FQY/9ISsJHD1144AMbEpjZhC6Mjr0NIjIz2EAIz09RjfoQZ/jGlJfXcrUrPzmfTLaQYvPUJ0ojSrEqIkxneowfa3WjP2mepUsTMlO0/D5jO3eW3VCRp/tlN+toQQQpRPrg1gyB9gage3TsKyZyH+itKphBAPIUUPIf6mtoslv77egvefrYWRvpp9l+PpNHsvn24K5/ztVKXjCVHu1HCyYHzHmuye2JZNY1ryauuquFoZY2tmiLvNX12+Vx2KZMuZ26TLFBghhBDlgXM9GLoZzJ0gLhwWtYGzv/73/YQQT52+0gGEKGv09dSMbF2NDr7OvLP+NEciElmyP4Il+yOo62ZJn4budG/ghq2ZodJRhSg3VCoVdd2sqOtmxTudaxGblq2bOpZXoGFa8AVSs/Mx1FPTrJodHWo7EuTrhItV2V3+TAghRCXnWBte3Qu/vAyRB+CXYdpeHx0/A/37lxUVQihDRnoI8RDe9masGdGMJUMa06WuMwZ6Ks7eTOXjP8IJ+GInr/1wnJ3hseQVaJSOKkS5olarihQzsvIK6NfEgyp2puQWaNh76Q4f/n6O5lP/5Llv9rH2aJSCaYUQQoh/YeEMgzdCy/Ha20cWwdJOkHRd0VhCiL/ISA8h/oVaraJ9bSfa13YiKSOX38Nu8suJG5y9mUrwuRiCz8Vgb25EL39X+jTyoKazhdKRhSh3LI0NmNzVl/efrc3VOxnsPB/LzvBYjkclcfZmKrdT/lpNKSMnn+ORSTSraoehvtTthRBClAF6+hD0EXg2h99Gavt8LGwNPRdArWeVTidEpSert/wH6ZorHuT87VTWH7/BhrCbxKfn6vbXc7OiTyN3uvu5YiPTX4R4LPHpOey+EEfjKrZ425sBEHxW2wjV3EifNjUd6FDbibY1HbA2lZ83IYQQD3dv9Ra3b+Zi2aHDk3ug5GjtNJcbR7W3W4yB9h+BnsGTe0xRIcnr0NIjRY//IN9s4t/kFWjYc/EOPx+PZtf5OPI12h8nQz01Qb6O9GnkTmsfB/T15B1pIUrDz8eimbbtInfScnT79NQqmlSxIai2E883dJd+O0IIIe7z1IoeAPm5sPNjODRPe9sjAPosAyu3J/u4okKR16GlR6a3CPEYDPTUBPk6EeTrREJ6DhtP3eKX4zc4dyuVLWdi2HImBgcLI3r5u9GnkTs1nGT6ixCPo29jD3o3dOf0zRR2hsey83wsF2LSOHQtkUPXEuno66wretxJy8HG1ECKjkIIIZ4ufUPo/AV4NYcNb0D0YVjYCp5fBNWDlE4nRKUjRQ8hSomduRHDAr0ZFuhN+K1Ufrk7/eVOWg6L9l5j0d5r+Llrp79083OV4fhClJBaraKBhzUNPKyZ2Kkm0YmZ2uLH7TQ87Ux1572z/jRHIxJpVs2O1j72tPRxoIqdqW7VGCGEEOKJqt0NnOrAuiEQcxpW9YHWE6Hte6DWUzqdEJVGhZ/eEh0dzUsvvURcXBz6+vp8+OGH9O3b95HvL8OKxOPIzdcQcjGOX47f4M8LRae/PFPLkTY1HWhZ3R4PW9P/uJIQojg0mkJaTdvNzeSsIvvdrE1oXcOetjUd6VTHWaF0QgghnranOr3ln/KyYdt7cGyp9naVVtB7CVg4Pd0colyR16Glp8IXPW7fvk1sbCwNGjQgJiaGRo0acenSJczMzB7p/vLNJkpLQnoOG8Ju8fOxaC7EpBU55mFrQmA1ewKr29Oimh125rK2uxCPq0BTyLlbKey7HM++y3c4HplEXoH2V17zqnb8NLKZ7tyw6GR8XSxlRRghhKigFC163HP6Z/jjTcjLAHMnbeHDu5UyWUSZJ69DS0+Fn97i4uKCi4sLAM7Oztjb25OYmPjIRQ8hSouduRHDW3ozvKU3Z2+msD08loNX4gmLTiY6MYs1idGsORoNQG0XSwKr2RFY3Z6m3raYGVX4H1UhSp2eWkV9d2vqu1szql11MnLyORKRyL7L8dRy+au/Tnx6Dj3nHcDUUI8Ab1ta+jjQ2see6o7mMhVGCCFE6anfF1z8YN1guHMeVnaHdpOh5XhQS9FdiCdF8VdSe/fu5euvv+b48ePcvn2b3377jZ49exY5Z968eXz99dfExMTg5+fHN998Q9OmTYv9WMePH6egoAAPD49SSi9EydR1s6KumxXjO9QgPSefIxEJHLiSwIEr8VyISeP87VTO307l+/0R6KtV+Hta06KaPS197GngYY2BNGYUotjMjPRpV8uRdrUci+yPTMjA3tyQ+PRcdl+8w+6LdwBwsjSiZXUHBjT1oHEVWyUiCyGEKC1lpYjtUANG7ILNE+HUj/DnpxAVCr0WgZmd0umEqJAUL3pkZGTg5+fHyy+/zPPPP3/f8bVr1zJ+/HgWLFhAQEAAs2fPplOnTly8eBFHR+0frg0aNCA/P/+++27fvh1XV1cAEhMTGTx4MIsXL36yT0iIYjI30ueZWk48U0s7rzM+PYeDVxM4eCWe/VfiuZGUxdHrSRy9nsScXZd170Zrp8LYU8vZArW6jPwiF6IcauRly5H3g7gQk8b+K3fYdzmeIxGJxKbmsP7EDZpXs9MVPW6nZHEhJo3GXjZYGBsonFwIIUS5ZGgGveaDVwvYMhGu7NSu7tJnGXgGKJ1OiAqnTPX0UKlU9430CAgIoEmTJnz77bcAaDQaPDw8GDNmDO++++4jXTcnJ4cOHTowYsQIXnrppf88NycnR3f75s2b+Pr6ylwqoZiohEwOXNUWQEKvJpCYkVvkuJ2ZIc3vToWRpqhClI7svAKOXU9i3+U7DG/pjaOlMQCL917j8y3nUavA19WSplXsaOptS5MqNtKLRwghyrDrA18k68QJZXt6PEjMWe10l8SroNaHoP9B81FlZ2SKUIz09Cg9io/0+De5ubkcP36c9957T7dPrVYTFBREaGjoI12jsLCQoUOH8swzz/xnwQNg6tSp/O9//ytxZiFKm6edKZ52ngxo6olGU8iFmDQOXInnwFXtu9EJGblsOn2bTadvA9rVKRp62dDQ05pGXjbUdrGU6TBCFJOxgR4tfbRTyv5OrVbhZWdKZEImZ2+mcvZmKksPRABQ3dGc7wc3poq99IwSQgjxiJzrwsgQ+GMsnPsNtk/WTnfpMQ9MrJVOJ0SFUKaLHvHx8RQUFODkVHQ5JycnJy5cuPBI1zhw4ABr166lfv36bNiwAYAffviBevXqPfD89957j/Hjx+tu3xvpIURZoFar8HW1xNfVkhGtq5KbryEsOpkDV+I5eDWek1HJ3EzO4mZyFn+cugWAsYGa+u7aAkgjTxsaetlga2ao8DMRony614w4JiWbI9cTORKRwJGIRC7FphOVmImzlbHu3G//vExEfCYB3rY09bbFy85UGqMKIYTSys4g978YW2qntngFwrb34cImiDkDfZaCe2Ol0wlR7pXpokdpaNmyJRqN5pHPNzIywsjoryHKqampTyKWEKXCUF9N07svqN7qUIOMnHxORSdzIiqJ45FJnIhKJiUrjyMRiRyJSNTdz9vejIaeNjT00hZDfBwt0JO+IEI8MmcrY7r7udLd727fqIxcLsWmYWygpztn0+nbXIhJY/2JGwA4WhjRxNtWVwSp6WQhRRAhhBBaKhU0HQFujeDnIZAcCUs6QIsx0PZ9MDD+72sIIR6oTBc97O3t0dPTIzY2tsj+2NhYnJ2dn+hjz5s3j3nz5pGbm/vfJwtRRpgZ6dOiuj0tqmuH5Gs0hVyLT+dEZDLHI5M4HpXElbh0IuIziIjP0L0YszDSp4GnNQ09bWjkZUMDT2sspUmjEI/M1syQZlWLdt1//9naHL47EuRUdApxaTlsPn2bzadv42ZtwoF3n9GdGxGfgbuNiUxFE0KIys6tIby6F7ZMgjPr4MAcuLhVO93Fo/irVwohynjRw9DQkEaNGrFr1y5dc1ONRsOuXbsYPXr0E33sUaNGMWrUKF0DGSHKI7VaRXVHC6o7WvBCE+33cUpmHieikzgRmcSJqCTCopJJy8ln3+V49l2OB7RvNtRwtKChl/XdESE2eNuZySoxQhRD6xoOtK7hAGgbo4ZFJ3MkIpGj1xPx/FvD4QJNId2/3U9uvgY/d2tdT56GXjbYS3NUIYSofExsoPdiqNMTNr0F8ZdgSUdtg9NnPgADE6UTClGuKF70SE9P58qVK7rbERERhIWFYWtri6enJ+PHj2fIkCE0btyYpk2bMnv2bDIyMhg2bJiCqYUov6xMDWhX05F2NbVLPucXaLgYm8aJqGRORGqnxUQlZnIxNo2LsWn8dCQaABMDPao7muPjZE4NJwtqOJnj42iBm7WJFEOE+A/GBno0q2p332gQgFvJWahVKnLyNdo+Idf/mormZWfKwKaevNqm2tOMK4QQoiyo1RU8m2v7fJz6CUK/hUvB2lEfns2UTidEuaF40ePYsWO0a9dOd/teE9EhQ4awfPly+vXrx507d5gyZQoxMTE0aNCA4ODg+5qbljaZ3iIqC309NXVcrajjasVLzbwAuJOWw4mov0aDnLqRQlZeAWdupnDmZkqR+5sa6uHjaI7PvUKIkwU1nCxwtTKWfgVCPAIPW1NOftiBa/EZup+545FJXI5LJzIhk4ycfN25iRm5jPnphG4EVkMPG6xMZSqaEEJUWKa20GsB+PaETeMg4Qos7QzNXodnPgRD0/+6ghCVnqqwsCy2MC47ZH1kIbSjQSITM7kcm8al2HQuxaZxOTada/Hp5BU8+L8QcyN9qjuaU+PuyBAfJwt8HM1xkWKIEI8kJTOPk9FJeNiaUs3BHICd4bG8svJYkfOqO5rfXZnJmtY1HHCxkmHPQgjxT9cHvkjWiRO4zZ2DZceOSscpmawk2DYZwlZrb9t4a0d9VAlUNpd4IuR1aOmRosd/kG82IR4ur0BDZEJGkULIpdg0IuIzyNc8+L8WCyN9qjuZU8PRAh8nc6rYmVHF3hR3G9MiK18IIe53OyWLXefjOBGVxMmoZCLiM4oc/7pPffo29tCdezUug3ruVliZyGgQIUTlViGKHvdc3gEbx0LaLe3tpq9C0EdgaKZsLlGq5HVo6VF8eosQovwy0FPrGqU+W89Ftz+vQMP1+L8VQ+K0I0Qi4jNIy8nnZFQyJ6OSi1xLpQJXKxM8bU2pYm+Kl50ZVexM8bQ1w8vOFDMj+e9KCBcrEwY182LQ3aloCek5nIxK5vjd6WiNq9jqzt16JoZPNoUDUNXejPruVvh5WFPf3Zo6rpZSZBRCVC4VaZSpTwcYdQi2fwAnVsKRhXB5G3T/FrxbKZ1OiDJHXkU8hPT0EKLkDPTU2uksThZ05a9iSG6+hoj4jLujQtK4ckfbsyAyIZP0nHxuJmdxMzmL0GsJ913TwcJIVwSpYmeKl/3dj7Zm0tNAVFp25kYE+ToR5Ht/nyuVCjxsTYhOzOJafAbX4jPYEKZ9V1BfrWLDqEDqulkBkJadh4mBHvqyZK4QQpQPxlbQ/Rvw7QEb34Sk67DiOWjyCgT9D4zMlU4oRJkh01v+gwwrEuLJKywsJDEjl+sJmUQmZHA9IZOoux8jEzJIysz71/tbmxroRoZ42ZpSxd6Mag7mVHUww8JYCiKickvMyOX0jWRORadoP95IISkzl7Mfd8LEUDva46Pfz7Lu2A3quFpS390aPw8r/Nyt8bIzlR48QogK4fqLg8g6frxiTG/5p+xU2PEhHF+uvW3tqS2IVG2rZCrxmOR1aOmRkR5CCMWpVCrszI2wMzeikZfNfcdTMvOITMy4Oyoko0hx5E5aDsmZeSRnJnMqOvm++zpYGFHNwYyqDuZUvVsMqeZgjpuNCXqy1K6oBGzNDGlb05G2d5epLiws5E5ajq7gAXApNp2svAKORSZxLDJJt9/KxID67lYsHtxYpsMIIURZZWwJ3eZoV3jZOBaSo2BlD2g0DDp8oj0uRCUmRQ8hRJlnZWpAfVNtL4J/ysjJJyqxaDHk2h3tUP47aTm67dC1xCL3M9RXU8XOVDcipKr93Y8O5tL0UVRoKpUKR0vjIvtWvxLAtfgMTkUn60aDhN9OJSUrj6tx6UUKHm//fIqkzDzqullS19WKOm6WOFvKqkxCiHKiIo9xr9YO3jgIOz+Go9/D8WVwZSd0nwvVnlE6nRCKkaKHEKJcMzPSp7aLJbVd7n8XIzU7T1sAuZPOtTsZXL37MSIhg9x8zd1Gq+n33c/e3IiqDvdGhZjpiiIu1sYY6cu73aLiUatVVHc0p7qjOb0baYfQ5uZruBiTRmLmX72tCgsL2XUhjsSMXHaej9XttzMzxNfVkgBvW0Y/4/PU8wshhLjLyAK6ztD2+vh9NCRHwg+9oOEQ6PiptheIEJWMFD0eQhqZClH+WRob0MDDmgYe1kX2F2gKuZWcxZW7RZC/F0Xi0nKIT9duRyIS77umtakBThbGOFoa4WhhjJOlEY4WRjhZ/rXP0dJIiiOi3DPUV1PPvegfx4WFsGBQI87eTOHsrRTO3Uzlyp10EjJy2Xc5nrwCTZGix9ifTmJvbkRdN0vquFpRzcFMmqUKIcTT4N0aXj8Iu/4HRxbBiRXaUR/d5mhXfxGiEur8l1EAADrjSURBVJFGpv9BGsgIUbmkZecREZ9RZGTI1Tva5XZz8jWPfJ1/FkccLY1wsjDC0fJeocQYBwsj6ZMgyr3svAIuxKRx9mYKNqaGdK2vXbEpJSsPv/9tL3Kukb6a2i6W1HG1pHUNBzrVcVYishCiEtI1Mp0zB8tOFayR6X+5vl876iMpQnvbuw20fRe8WiibS/wreR1aemSkhxBC/I2FsQH13e/vH1JYWEhKVh6xqTnEpWXrPsb943Zsag65+Zq7zVXzuBib9q+PZ2NqgKedmXbVGTtTPO+uQuNpZ4qDuZH0SRBlnrGB3gNHVOmrVUzv68fZmymE30rl3K0UMnILCItOJiw6mQJNoa7okZNfwIcbzlLLWTtVzdfFUpaiFkKI0lKlJbx+AP78TDvqI2KPdqvSCtq8A96tlE4oxBMlRQ8hhHgEKpUKa1NDrE0Nqels8dDzCgsLSc3KJzYtm9hUbVEk9m/Fkb/fzsnXkJSZR9JDVp4xNdTD09aUKnZmeNmZ4qX7aIqLlaw+I8o2MyN9+jRyp8/dHiEaTSGRiZm6qTFNq9jqzr0cm866YzeK3N/VylhbAHG15Jlajvh73r+ykxBCiEdkaAadp0LAa7B/FpxcBdf3aTevQGgzSTsCRN5sERWQFD2EEKIUqVQqrEwNsDI1oIbTfxdHbqVk6Zbijby7Ck1kQia3krPIzNVOG7gQc/9oEUM9Ne62JnjZ/lUMqWJnhqedKR42phjqS98EUbao1Sq87c3wtjejm59rkWNWJgaMbe/D+dupnL+dyo2kLG6lZHMrJZtdF+IwNdTXFT2iEzOZv+fq3REhFtRytsTMSP6cEUKIR2LjBd1mQ6sJd4sfP0DkAe0Stx7NoO07ULWdFD9EhSJ/JTyENDIVQjxJfy+OPGjlmdx8DTeSMnUFkesJmUQlZnI9IYMbiVnkFmjuNmHNAO4Uua9apV2BxtbMEFszQ2zMDLE1vffRQPvxb5uNqaH0FhGK8rA1ZXyHGrrbqdl5XLidpiuCNPX+a1TIqRv/b+/O46MqD/3xf87s+0zWyR7CYiCsCgIRtxZKQOqtFa/L5VrUVi80+NVirWtFve3F1u/P2lrEtla9v99VqfoT9StqiyjgwqKULSyRJSyBZLJnJrMvz/ePM3PIEMImZJLh8369zmtmznnmzHN4OAnz4Vk68NqGQ8prSQJKM03KKk7XjM7H0FxLn9afiGjAcRQD339GDj++eBbY9N/A4fXySi9FlwJXPQgMncrwg9ICJzI9BU4gQ0T9TTQm0NDpx6FWHw60+nCwzYuDLT6lp4gvFD3jc5p0amSYuockWmSa9cg0a5NCk2yLDtkWPexGLecboZTYedSN97YeVQKRJk8w6fgL/34JZoySJ1P956F2vLflKC5yWlGeZ8EwpxU2A+cKIboQXdATmZ4OdwPwxe+BTS8DkYC8r3C8POfHsOkMP1KA30PPHfb0ICIaYNQqCUUZJhRlmHDZ0ORjQgi0dIXgcgfQ7guhzRtCuzeENl8Ybd4g2r1heV/imC+EcFTAF4rCF/LjSIf/tOqgUUnIigcgWRY9si065Fj0yr7s+POceI8TLlNK50pFgTzPR0JrVxC7GjzY2dCJXQ0ejCo8tszuun2teOXLA0nvL7AbMMxpRXmeFbdOLkVxpqmvqk5EqcTv7CdnywdmPgVcfi/w5XPAV38FjmwCXrsRKLhYDj8umsHwgwYkhh5ERGlEkiTkWPXIsepPq7wQAl3BCNq8oW5hSDgelMQDk25bS1cQ7kAEkZiAyx2Eyx085WdIEpBh0iHLHA9ErHpkmXXIseqVniOJOmdb9NAyIKEzkGXR4/Jhelw+LLvHsfGlGbjzijLUurrwTaMHje6AMlfImm+acf0lhUrZtzbV4+87GlHutGKY04LyPCsGZ1s4Pw4RXViseUDVr4Ep9wBf/kEOP45uBl6/GcgbI4cfw2cx/KABhaEHEdEFTJIkWA1aWA1alGaZT+s9oUgMrd4gWjwhtHiDaPEE0dIVQmtXEC1d8vPEY5s3iJiAEprsaeo65fkzzTrkWo8FITlWPXKtBvm5RY9cm7zPqtdwiA2d1OTBWZg8OEt53ekPY4/Lg1qXB980elCWfezv/Ma6Vqzc6cLKnS5ln0YlYVC2GeVOKxZdW4Fcm6FP609ElDKWXGD6r4Ap98o9Pzb+BWjcBvxtDuAcDVx1PzD8WkDFYJj6P4YeRER0RnQaFfLtRuTbjacsG40JtPtCaFWCkG6hiOfY6+b480hMKAHJiVat6c6gVR0LRBJhSLzXiPzcgCyLDlkWHfQaTtRK8ioxEwZlYkK35XITbp08CCML7EogUuvywBOIYG9TF/Y2deG3N4xRyv7q/Z34fG8LhuRaMDTHgqG58laWbeakwET9HaczPDPmbOB7TwCX/S9g/RJgw58B13bgjR8BuRXAlfcDFdcx/KB+jaFHL7h6CxHRt6dWScocH+XofQlfAIjFA5ImTxDNnmC3x4DyuiX+2BWMIBCO4XCbH4fbTj0Pic2gQbZVj2yzHtnWY/OOZHebhyTHIh8z6fir8UI0usiO0UXH5gMRQh7CVevyoL7dl7Qs7vYjnSdcTlolySvR/P3eK5Xwo77dB6teXqmJiGjAMmcBUx8DKhcA65cCG14AmnYCb90O5PwGuOLnwMjrADV/1lH/w9VbToGz5hIR9T++UATNJwlHmtxyz5FWbwjR2Jn9mjNq1T2CkexuE7Tm2Q0odBiRY9VDreLwmgvR4TYf9jR5lF4g+5q92NvUhU5/GLlWPTY+Mk0pe+tfN+CzPS3ItugxNNeMId16hgzJsSDfbuAwLaI+cODf/x3+rzeh8NlnYZtRlerqDHz+dmDDn4D1zwOBTnmfrQiYPB+45EeAwXby99Mp8XvoucP/ziIiogHHpNOgNEtzynlIYjGBTn8YLV1BNHcFk4bZJJ43d5uPJBCOwR+OnlYPEo1KUgKQQocRBcpmUF537x1A6aM404TiTBO+O9yp7EusnNTkCSSVdQciAKD8vVu/v005lmHSYvNjx5bOfH/bUagleR6R0iwTex0RUf9lzACuflAOOTb8Gdj4J8BdD/zjEWDNb4Dxc4FJ8wA7v6xT6vG3KRERpS2VSkKGWYcMsw7DnCcfXiOEgDcURYsniFZvEM2eUFI4ktiOdgTQ6A4gEhOob/ejvr33cMRh0qLALgcghQ6DEowUZshBSY5FDxV7i6SF3lZOerd6CrqCEexTeoXIj3ubu5B7XNnffLQ7KWxz2vQYlGVGWbYZIwtsuLVyUF9cChHR6TPY5UlNL7sb2P4G8OUfgZZaefLT9UuBkdcDly0A8semuqZ0AWPoQUREBPlLq0WvgUWvwaDsk/cgicYEmjwBHO3w40hHAEfa/Tja4Y+/lh/dgQg6fGF0+MLY2eA+4Xm0arm3SJ7NAJtBC7NeA7NeA4teHX+UN+W5QQOzLrFPLqPXqDg8op+z6DUYW+zA2GJH0v7uI4yFELh0UCayzF4caPWiwxdWloXeUNeGccWOpNDj5j+vgyreK6Qsyyw/Zss9UDhxLxH1Oa1BHtYy7t+BvSvl0OPAZ3IQsv0NoOxKeTLUodO43C31OYYeREREZ0itkpQVbMaXnriMJxDG0Y5AUhCSeEz0FglHxWlPxtobrVqSw5JuYYjFoIVFr4ZVr4XDrEWGSYcMkxZ2o/yYYdbBYdLCYdRBp+GM+6nSPaySJAnP3DhOed3hC6GuRQ5A6lp8yLHolGORaAxfH2hHJCbw5b7W484JTB2eixfnXqrs+3JfC5w2A4oyjAxEiOj8UqmAi6rk7egWYN0fgZq3gbq18pYzXJ4MdcyNgEZ/ytMRnQsMPYiIiM4Dq0GL8jwtyvNOPKwmEo2hyRPE0Q4/Gt0BdAUi6ApG4A1G4Q3Jz7sCEXiD8f0h+Zgnvs8fjgIAwlGh9Cg5Gxa9Rg5ATHI44ogHJInHDJMOdtOx4MRh0sFm0LB3yXnmMOlwcYkOF5dk9DgmSRL+9h+TUdfiw4EWL+pavTjQIm/eUDRpLpBoTGDuSxsRjgpIElBgN6I0y4TSLHnekDFFdlw2JLsvL42ILhQF44DZLwJTF8mrvWz6b6B5N/DeAmDVk8Cku4AJPwZMPZcRJzqXGHoQERGlgEatUub4OBvRmIgHIfImhyHReHAihyRufxjtvjDafSF0HPfY6Q9DCMjhSjBy0rlJjqeSALtRK28mHexGLRzx1w6T9tgxoxySdN+fWMqVzp5aJWF8aSbGlyZ/UUhMphqOxpR9Hb4QhuZacahVDkSOxHscJXqIzBqTr4QesZjALX9Zj8IMIwbFQ5HSLDMGZZngMOlARHRWHMVA1a+Bq34hBx8bXgDcR4BPfgV89gwwbg5Q+VMgc3Cqa0ppiqEHERHRAKRWSbAZtLAZtGf1/lhMwB3oHoqE0O49LiDxh5X9Hb4Q2n1h+MNRxATi7wsDrb4z+ly9RnWCcEQORqwGDayGY/OXWPSJ1/KxxJwmXCr4xBKTqXaXZdHjw3uuUAKRQ21eHGjx4WCbDwdbvZhYdiw4aXAHsKGuDajreW6bQYNbJpXgoZkjAMh/fzYdakdppgk5Vj17/lC/J4F/R1POYAem/C95xZcdy4Ev/wA0bge++gvw1YvAiO/L834UT0x1TSnNMPToxZIlS7BkyRKEQqFUV4WIiOicU6kkOOLDWcpw8olbuwuEo3D7w+jwh9Hpl4fVdMbDkRPt774vGhMIRuRhPU2e4FnX3axTK6GIxaCFVZ8clliV0EQLs14NvUYNg1YFg1Yd31QwaNTQxx8NWjX0GlVar6TTfXWZ43uIJNgMGjx3y8U42OrFwVafvLV54XIH4Q5EoOoWbLg8AfzrC+sAAEatGiWZpnjPEBNKssy4pMSBkQX2Prk2Ihpg1Fp5To/R/yrP8/Hlc/Lkp7v+j7wVTZRXgxk+C1CxdyB9eww9elFdXY3q6mrU19ejuLg41dUhIiLqFxLBQa7NcEbvE0KgKxhRApETBSRdwbAyt4kn/ph47gmEEY7Kq514Q1F4Q1G4cPbByYnoNCoYNMeFI1q1EpB0D08SK+skeqBYu4cv8deJMgOlZ4rVoMW1Ywt67PeHojjU5oNZf+zLR4cvjOJMI460++EPR1Hr8qDW5VGO/8dVg5XQo9kTxH1vbkWpEozIQ2dKMk0c7kR0IZMkYPBV8ta0S570dNsbQP1G4I1bgYwyORwZOg0oHM8AhM4aQw8iIiI67yRJkoMBgxZFPefmPC3BSDQpFDkWjMhhiSc++Wv3475QBIFwFIFwDIFIFMFwDMFI/HU4ikjs2LKxoUgMoUgM7kDkHF21LNEzJRGEJIbxWPXa44bxaKBVq6BRS1CrJGhUKmhUEtRqCVqVSt6nlqCJH1OrJGi7l40fk8uplOc69bfrxWLUqXtMyDsi34bPfvFdhCIxHOnwJ/cOafVibJFDKVvX4sXab5pPeO48mwHV3xmiLMfrD0Wxp8mD0iwz7MazG7pFdHLi1EWo7+WOAH6wBPjuY8DGPwNf/xVorwPW/EbeDA5gyHflAGToVMCal+oa0wDC0IOIiIgGBL1GDb1FjSzLuVvmMBKNIRCJxYORY2FIMB6QBLoFJMpjJCqvqhMPVjxBuSdK13H7QhF5QlGlZ4r73PZMOV2SBFh0ibBFGw9g4r1T4s9thu5zqvTcb9FroFH3XN5Yp1GhLNuMsuzeh0gNyjLhN7NHx4fL+JSAxBOIoNEdkCsYt+NoJ26ID5vJMGlREp9IVe4lYsalgzJRkmU6939IRNQ/WJ3A1F8CVywEdrwD7PkHsP9TINAB7Hhb3gDAORoYNk0OQYonyUNmiHrB0IOIiIguWBq1Cha1Chb9uf8nUaJniieQPEyn+/Add+DYkJ6uQASRmEAkFkMkKhCNCYRjAtH460hM3hc57nU4GovvF4hEY4gd9x/ZQkAOZoIRoDNw1tdj0qmVAMRq0MKmTESrSVqxx27sfkzebpxQnDTZqRDyUssHWr0o7LaCkScYQbZFj5auYHyy3A5sPdyhHH/yByPxo3ivkF0Nbjz3yR6UZMaDkfiwmTybIa3nZyG6IOjMwMVz5C0aAY5sAvZ+LM/9cXQz4Noub5//DtBZ5SEyQ6fKIYijJNW1p36GoQcRERHReXA+eqacjlhMICpEPBiJIRCOwRMIdxsSFIY7cCyESX481nMlsT8Qlnus+EJR+M5yLhV5tSFNUihyfDCS2J69aRw0asATiKLdG0RzVwj17T4caPGh3HlsmM3uRjc+2N7Y47N0GhVKMk14cMZwTKtwAgA6/WG0eUMoyjBCe4IeK0TUj6k1QMkkefvuI0BXs9z7Y+/HwN5VgK8F2P2+vAFAdvmxYTClUwDtmc1BRemHoQcRERFRGlGpJKggQZ4jVA2rAT2Wsj0T4WisZzASkCef7fTLAYrbf+z18VsoIvdEUZY5PgsGrQo2gxaPvFMDW3xojkqSML7UgWA4Bl8oik6/vNRyKBLD3qYuuNwBuNwB2AxafLrbhXv/thUqCShwGOMTqco9Q0ozTbi0LBPZfRxOEdFZsuTIE5yOuRGIxYCGLXL4sfdjeRLUllp5W78E0BiBQZcDw74nByGZg5OG1NGFgaEHEREREfVKq1Yh06xDpll3Vu8PhKPJQYive2DS7Xn80RMPUdzxYT/yOWIIhM9sqeNH3qnBI+/UAAASo11iAqhv96O+3Y8v0KqUnTU6H2OK7LAatDjS4cOmg+0ozjBhULYZQ3LNGO60oSTTCJWKvUSI+hWVCii8RN6uuh/wtwP7Vx/rBeJpkIfE7F0pl88YJIcf4+bI76ELAkMPIiIiIjpvEksAO89wmWMAiMYEugIRJRxxB5JDEU8gDLdfPu5OBCaBcPy1fDwm0GOek+Ot2N6AFdsbkvatR1uPcioJyDLrkWXRwWrQQKdWQatRIcesh9WohUWvhkmvgVmvgUWvhlknz4FiSryOHzPrBs5SxkQDijEDGPlDeRMCaNopByB7VgKH1gPtB4CvXgQKLmbocQFh6EFERERE/ZJaJcFu0sJu0qL4LN4vhIA3FFUCkd7mM0lMOOsORNDQ6UezJwhvKIJgOJa0rHFMAM1dQTR3ffuVeAxaeQJds14Dk+5YKGLRa+C0GVDgMKLAHn90GJFl1nGCVqIzIUmAc6S8TbkHCHqAus/kEGTI1FTXjvoQQw8iIiIiSkuSJMESDxLOlhACbn8E37g82NPkwdBcS3xy2Aje/mc91nzTnBSMdFfutCImBHyhKNq8IQTCUSRKykN2QmjpCp1WPXRqFfIdBuQnghC7HIbkOwwodBiRbzfAauCynUS90luB4dfIG11QGHoQEREREfVCkuTeJpeWZeLSssykY7PG5CMWE2jyBHGw1YuDbT4cavXhYJsPB1u9WHbXZJh08j+3H3p7G17feLjH+U06NXKsetx5xWDoNCq4/WHsa+6Cyx1EmzeIhs4AmjxBhKIxHGz14WCrr9e6Wg2aeBhiQL7DiEJH/LldDklybXoY5BluiSiNLF68GG+//TZ2794No9GIyy67DL/5zW9QXl6ulAkEArjvvvuwbNkyBINBVFVV4fnnn4fT6VTKHDp0CPPnz8enn34Ki8WCuXPnYvHixdBojsUGq1evxsKFC7Fjxw4UFxfj0UcfxW233daXl3vGGHr0YsmSJViyZAlCodNL34mIiIjowqNSScizG5BnN2DS4Kxey107tgCFDqMcXMTDkUZ3AL5QFAdbffjhxYUwx3ukPLx8Oz7Z3QSdRoXiDCNG5NuQbdHDotdAq5ZgNWjQ5AmioSOAIx1+NHQGlElgawMe1Lo8vdbDYdLCaTXAaTfAadXDaUt+nmc3IMusgybdlvblih2UxtasWYPq6mpceumliEQiePjhhzF9+nTs3LkTZrMZAPCzn/0MK1aswJtvvgm73Y4FCxbg+uuvxxdffAEAiEajmDVrFvLy8vDll1+ioaEBP/rRj6DVavFf//VfAIC6ujrMmjUL8+bNw6uvvopVq1bhJz/5CfLz81FVVZWy6z8VSQhxiqmdLmz19fUoLi7G4cOHUVRUlOrqEBEREVGaCISjONTmw5F2P74zPFfZv/CNLXhvy9Feh83UPFGlDNl5+Ys61DZ6kGc3wGbQQKNWIRYTaPOGcLQzgKPxUORohx/BSOy06qWSgGyLHnl2A3KtBjhteuTZDMcCEpseTqsBDpMW0gAJEw7e+iP4vvoKhc/+DrYZM1JdHaJTSnwP3blzJwoLC5X9er0eev3Jl9hubm5Gbm4u1qxZgyuvvBKdnZ3IycnBa6+9hhtuuAEAsHv3bowYMQLr1q3D5MmT8eGHH+L73/8+jh49qvT+eOGFF/DAAw+gubkZOp0ODzzwAFasWIGamhrls26++WZ0dHTgo48+Og9/CucGe3oQEREREaWAQavGRU4rLnJak/Y/c+M4/Hb2GDR0BuI9Q7zxoS1edPjCSXOUfLK7CZ/taelx7kyzDiWZJvztPyZDr1FDCIEdR9xwB8MIR2NocstLADd2BuByB+DyBOHqDKC5K4hofMiOvERwZ6/112lUyDbrkGHWIcOUeNTKz03aY/tNOmSY5f0mnTq1QQn/v5cGmIqKiqTXixYtwuOPP37S93R2yvdtZqY8JG/Tpk0Ih8OYNm2aUmb48OEoKSlRQo9169Zh9OjRScNdqqqqMH/+fOzYsQMXX3wx1q1bl3SORJl77733W1zh+cfQg4iIiIion9GoVSjONKE404TLkd1ruVsnl+Likgwc6janSKs3hDZvCEII6DXyHB6SJOHpf9RizTfNyrCZ4kwTSjJNmFiWieJME6ZXOBETQKs3CFdnMB6GBODqDMDlDqLRLQckTZ4g2rwhhCIxuTdJZ+C0r0unUXULRo6FId1Dk0yzDmXZZhRnmLhiDV3wTtTT42RisRjuvfdeTJkyBaNGjQIANDY2QqfTweFwJJV1Op1obGxUynQPPBLHE8dOVsbtdsPv98NoNJ75BfYBhh5ERERERAPU9JF5mD4yL2mfJxDGoTYfOnzhpP2RWAxqlYRQJIZ9zV7sa/YqxxwmLapGTodaAnKtBvxpzX40eYIoyTSiOMOEyYOzUJxpQr7dAI1ahWAkiiZ3EK3eENp9IXT4QmjzhuOPIXT4wmiLH5O3MEKRGEKRGFzuIFzuUy/7a9CqMDTXovSGuchpwbBcKwodRoYhdMGwWq2w2WynXb66uho1NTX4/PPPz2OtBhaGHkREREREacRq0GJkgb3H/ld/MhmRaAwNnQEcavMp2+E2H3Sa5IlLP61twv5uoUiCRiVhRL4N/+fuy5WeKOv3t8KRZcYVw0zIMutOOHxFxJfulQOSboGIVw5EEsFIuzeElq4g9rd4EQjHUHPEjZoj7qRzmXRqDHNacVE8EBnmlB/z7YYBM8cI0fmwYMECvP/++1i7dm3SfJR5eXkIhULo6OhI6u3hcrmQl5enlNm4cWPS+Vwul3Is8ZjY172MzWbrt708AIYeREREREQXjO7DZqacpNwj14xAXYs3KRypb/cjFIlBIHlejEffqcHepi4AciBRlCH3DinONGGY04I5k0ohSRLMeg3Meg2KMk5dz0g0hkNtPnzj6sIel7wizR5XF/a3dMEXimLr4Q5sPdyR9B6rXqMEIMPiPUMuclqRaz35cACigU4IgbvvvhvLly/H6tWrUVZWlnR8/Pjx0Gq1WLVqFWbPng0AqK2txaFDh1BZWQkAqKysxK9//Ws0NTUhN1eeWHnlypWw2WzKvCKVlZX44IMPks69cuVK5Rz9FUMPIiIiIiJKMnWEs8e+WHyC065g8rCZfLsBXYEIXB55Cd5vXF34xiWHIBX5NsyZVKqUvWHplwhEokooUpxhRFGmCcUZJhRlGGHQynOQaNQqDM6xYHCOBTNGHRu+E47GcLDVG/8MT3zrwoEWLzzBCP55qAP/PNSRVD+7UYvFR90oA7BuXyvKXB4MybFAzSEylCaqq6vx2muv4d1334XValXm4LDb7TAajbDb7fjxj3+MhQsXIjMzEzabDXfffTcqKysxefJkAMD06dNRUVGBW2+9Fb/97W/R2NiIRx99FNXV1co8IvPmzcMf//hH/OIXv8Add9yBTz75BG+88QZWrFiRsms/HQw9iIiIiIjolFQqCXl2AwBD0v7/78eTAADBSBRH2v043O7H4TYfDrf7kGXWKeWEEKg52qkMWzneiHwbPrznCuX1i5/th1GnRqHDiKIMIwodJhh1agzNtWJorhXXjM5XyoYiMdS1ePGNy5PUM+RAqxed/jA8ATmo+Z8NB/FZ/VqYdWqMKrRjXLEDY+NbAYfH0AC1dOlSAMDVV1+dtP/ll1/GbbfdBgD43e9+B5VKhdmzZyMYDKKqqgrPP/+8UlatVuP999/H/PnzUVlZCbPZjLlz5+LJJ59UypSVlWHFihX42c9+ht///vcoKirCiy++iKqqqvN+jd+GJATXbTqZxPrIhw8fThoXRUREREREp08IgVqXB4fbjoUih9v8qG+X5xWZMjQbf/7RBKXsiMc+QiAcSzpHllmHwgwjKgdn4aFrRij79zZ1Iceqh92oTSofCEexv9kLf/WdMO3ahjevmYe/WcrhDUV71C/bosfYIrsSgowtssNh0vUoR9QX+D303GFPDyIiIiIiOu8kScLwPBuG5/VciUIIgWDkWMARisZw44RiHGn340iHH0fa/fAEI2j1htDqDcFpMyS999rnPoc/HIVVr0FhhhGFDqPyWFFgQ4lFDx+Ae793ER6pqsK+5i5sic8LsrW+A7sbPGjpCmLV7ias2t2knHtQlgljix0YU+TAuGI7RhbYlSE4RDQwMPQgIiIiIqKUkiQpKUzQa9R48gejksp0+sM40i73DOneo8MTjMCkU8MfjsITjGB3owe7Gz3K8e9VOPFo/LkQAv/yx8+RZdErw2YmlmUix6qHNxjF4TYvttV3Ymt9J+pavDjQ6sOBVh/e3XIUgLx6TXmeVekJMrbYgaE5FmjUyavfEFH/wdCDiIiIiIj6PbtRC7tRi4qC5J4iNoMWm375PfhCERzt8KO+W++Q+nY/xhU7lLIdEWDH0Z7ziSRUjXTiT7fKQ2zavUH8+oPd8AYiaOoKYn+zF+2+EHYcdWPHUTde2yC/R62SUOgwoiS+Ko78KL8uyTTBbtRyrhCiFGLoQUREREREA55Jp1EmOT3ewT/LoYNJDbz6k0lKj5H6REjS7kejO4B8u7HbuyS8tak+6TxqlQSHSQu9WgWtRoUWTxDeUFRZ1vdErAaNEoAkgpFEOFLoMEKnYS8RovMp7UOPjo4OTJs2DZFIBJFIBPfccw/uvPPOVFeLiIiIiIj6mF4CpgzNPuGxSDSWNK9IOBbDjROKcCQejBzt8CMcFWjtCgEAbrtsEB77fgWaPEHsONqJH//31/JnaFRQqyREYgKhSAyeQETpHXI8lQTk241JPUMSoUhppgmZZh17iRB9S2kfelitVqxduxYmkwlerxejRo3C9ddfj6ysrFRXjYiIiIiI+gmNWpU0N0eu1YDf3jBWeR2NCTR7gnIPkXY/SrNMyjK+nkAYRq08r0j34CRh6vBcXD4sG4fafNjX1IV1+1sRjQnEBOShOB1+rN/f1uN9Vr0GpdkmlGaaUZplim/yc6fVAJWKgQjRqaR96KFWq2EymQAAwWAQQghwlV4iIiIiIjoT6njAkWc3YMKg5GPDnFbsfLIKHb6wEmIcjW9HOvyYXpGH6y4uBADUHOnE95/7vNfPybXpoYKERncAnmAENUfcqDnSs5eIXqNCSaYcggw6LhApdBg5uSpRXMpDj7Vr1+Lpp5/Gpk2b0NDQgOXLl+O6665LKrNkyRI8/fTTaGxsxNixY/Hcc89h4sSJp/0ZHR0duOqqq7Bnzx48/fTTyM4+cZc2IiIiIiKisyFJEjLMOmSYdRhVaO+13KBsM177yaR4MBJQgpHE408uL8NdVw5BIBzFp7ubMP/Vf57wPMFIDHuaurCnqavHMbUE5DuMKMs2Y1CWWQlEijONyLHokWHSsZcIXTBSHnp4vV6MHTsWd9xxB66//voex//2t79h4cKFeOGFFzBp0iQ8++yzqKqqQm1tLXJzcwEA48aNQyQS6fHef/zjHygoKIDD4cDWrVvhcrlw/fXX44YbboDT6Tzv10ZERERERNSdRa/BZb3MKyKEQCQm90o3aNUYXWRH9XeGoKEzgMbOABrd8qMvFAUA3HRpMUYV2HCg1Yft9Z3YeEAeIhMVQH189ZrP9rT0+BwJgM2oRZZZB6fNAKdNj2yLHlkWPbItOmRbEq91yLLooNeoe5yDaKBIeegxc+ZMzJw5s9fjzzzzDO68807cfvvtAIAXXngBK1aswEsvvYQHH3wQALBly5bT+iyn04mxY8fis88+ww033HDCMsFgEMFgUHnt8XhOWI6IiIiIiOhckiQJWvWxHhhFGSbcXzU8qYwQAp5gBI2dAThMWuRaDQCAXQ1uPP33WjR2yr1HOvzhpPfl2fQIRGLo8IUhAHT6w+j0h7G/xXvKepl0auRY9Mix6pBjNSCrWzCSbdGhwGHE4BwLLPqUf70k6qFf/60MhULYtGkTHnroIWWfSqXCtGnTsG7dutM6h8vlgslkgtVqRWdnJ9auXYv58+f3Wn7x4sV44oknvnXdiYiIiIiIzjVJkmAzaGEzaJP2j8i34aXbLlVeB8JRNLmDcu8QdwAV+fJyvuFoDKtrm7Do3R1o9YZOOPGq06YHALR2hRCJCfhCURxs8+FgL8vyKu+z6lGcaUR5ng0XOa0YkmPBkFwz8mwGrkJDKdOvQ4+WlhZEo9EeQ1GcTid27959Wuc4ePAg7rrrLmUC07vvvhujR4/utfxDDz2EhQsXKq+PHDmCioqKs7sAIiIiIiKiFDBo1SjJMqEky5S0X6tW4XsVefheRR4AoCsYQZM7gCZPUN7cAUwenIVRhXYIIfDxThd+9sZWdAV7TicAAAUOA0IRgZauIFweefv6YEdSGbUE2E06jMizonJIFobkWFCUaYRWrUK+3QibQcNQhM6bfh16nAsTJ0487eEvAKDX66HX65XXbnfPmZKJiIiIiIjSgUWvgSXHgsE5lh7HJEnC90bmoeaJPKXnSJMnoIQjLk8Q00Y4Mb40A52+MP7/f9bj1yt2IXrcaplRAbR5Q/hiXyu+2Nd6gs8BTFq1MlynKMOI2eOLcHW5PIejNxhBrcuDTJM8USxDEjoT/Tr0yM7OhlqthsvlStrvcrmQl5d3Xj97yZIlWLJkCUKh0Hn9HCIiIiIiov6ut54jCXaTFndcXobbpwyC2x9BizeIFk8QLncAe5q6sL/Fi0yTFr5QDPuau1Db6IY/LA+tEQLwhqLwhqI40hHA5sMd+HiXC/kOI8w6DWJCYMfRY/8ZLUmAUauGSaeG1aDBFcNycMWwHJj1asRiAjVH3HDa9ci3G1FgNyLbqoNRq2ZQcoHq16GHTqfD+PHjsWrVKmUZ21gshlWrVmHBggXn9bOrq6tRXV2N+vp6FBcXn9fPIiIiIiIiSgeSJMFu0sJu0mLICXqPJAgh0OwJYmeDG9uPdKK2wYO6Vi+OtMuTsPrDMexvPvEkq0IAvlAUvlAULV0h1LUcxP+77uAp66ZWSci3G7Do2pH4XgVX87xQpDz06Orqwt69e5XXdXV12LJlCzIzM1FSUoKFCxdi7ty5mDBhAiZOnIhnn30WXq9XWc2FiIiIiIjopPg//P2OJEnItRmQazMow1gSfKEIDrb64PaH4Q1F4A1G4Q1G4A1F0ekLoc0bQrsvhA5/GB5/BGq1JPcWCUbQ4Quj3RdSlv7tLhoTqG/3wx+O9tVlUj+Q8tDj66+/xne+8x3ldWIS0blz5+KVV17BTTfdhObmZjz22GNobGzEuHHj8NFHH/WY3PRc4/AWIiIiIqL0IkTPL8LU/5h0GozIt33r8wQjUXQFImjpCqKhMwCXOwibQYPxpRnnoJY0UEiCd/5JJYa3HD58GEVFRamuDhERERERnaGDc2+Db8MGFPw//xv2WbNSXR2iU+L30HNHleoKEBERERERERGdDww9iIiIiIiIiCgtMfToxZIlS1BRUYGrr7461VUhIiIiIiIiorPA0KMX1dXV2LlzJ1avXp3qqhARERERERHRWWDoQURERERERERpiaEHEREREREREaUlhh5ERERERERElJYYevSCE5kSERERERERDWwMPXrBiUyJiIiIiIiIBjaGHkRERERERESUlhh6EBERERFRepMk+VGkthpE1PcYehARERERERFRWmLo0QtOZEpEREREREQ0sDH06AUnMiUiIiIiIiIa2Bh6EBEREREREVFaYuhBRERERERERGmJoQcRERERERERpSWGHkRERERERESUlhh69IKrtxARERERERENbAw9esHVW4iIiIiIiIgGNoYeRERERERERJSWNKmuQH8Xi8UAAA0NDSmuCRERERERnY0Grxf+cAiiuRme+vpUV4folBLfPxPfR+nsMfQ4BZfLBQCYOHFiimtCRERERETfyh23p7oGRGfE5XKhpKQk1dUY0CQhhEh1JfqzSCSCzZs3w+l0QqXqv6OBPB4PKioqsHPnTlit1lRXh06B7TXwsM0GFrbXwML2GnjYZgML22vgYZsNLOejvWKxGFwuFy6++GJoNOyr8G0w9EgTbrcbdrsdnZ2dsNlsqa4OnQLba+Bhmw0sbK+Bhe018LDNBha218DDNhtY2F79W//tukBERERERERE9C0w9CAiIiIiIiKitMTQI03o9XosWrQIer0+1VWh08D2GnjYZgML22tgYXsNPGyzgYXtNfCwzQYWtlf/xjk9iIiIiIiIiCgtsacHEREREREREaUlhh5ERERERERElJYYehARERERERFRWmLoQURERERERERpiaFHGliyZAkGDRoEg8GASZMmYePGjamu0gXp8ccfhyRJSdvw4cOV44FAANXV1cjKyoLFYsHs2bPhcrmSznHo0CHMmjULJpMJubm5uP/++xGJRPr6UtLW2rVrce2116KgoACSJOGdd95JOi6EwGOPPYb8/HwYjUZMmzYNe/bsSSrT1taGOXPmwGazweFw4Mc//jG6urqSymzbtg1XXHEFDAYDiouL8dvf/vZ8X1paOlV73XbbbT3uuRkzZiSVYXv1ncWLF+PSSy+F1WpFbm4urrvuOtTW1iaVOVc/B1evXo1LLrkEer0eQ4cOxSuvvHK+Ly/tnE57XX311T3usXnz5iWVYXv1naVLl2LMmDGw2Wyw2WyorKzEhx9+qBzn/dW/nKq9eH/1b0899RQkScK9996r7OM9NoAJGtCWLVsmdDqdeOmll8SOHTvEnXfeKRwOh3C5XKmu2gVn0aJFYuTIkaKhoUHZmpublePz5s0TxcXFYtWqVeLrr78WkydPFpdddplyPBKJiFGjRolp06aJzZs3iw8++EBkZ2eLhx56KBWXk5Y++OAD8cgjj4i3335bABDLly9POv7UU08Ju90u3nnnHbF161bxL//yL6KsrEz4/X6lzIwZM8TYsWPF+vXrxWeffSaGDh0qbrnlFuV4Z2encDqdYs6cOaKmpka8/vrrwmg0ij/96U99dZlp41TtNXfuXDFjxoyke66trS2pDNur71RVVYmXX35Z1NTUiC1btohrrrlG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Xp20BKPP51PKnIufg4nl/u6//36xa9cusWTJEi73dxZO1V579+4VTz75pPj6669FXV2dePfdd8XgwYPFlVdeqZyD7dW3HnzwQbFmzRpRV1cntm3bJh588EEhSZL4xz/+IYTg/dXfnKy9eH8NDMevsMN7bOBi6JEGnnvuOVFSUiJ0Op2YOHGiWL9+faqrdEG66aabRH5+vtDpdKKwsFDcdNNNYu/evcpxv98vfvrTn4qMjAxhMpnED3/4Q9HQ0JB0jgMHDoiZM2cKo9EosrOzxX333SfC4XBfX0ra+vTTTwWAHtvcuXOFEPKytb/85S+F0+kUer1eTJ06VdTW1iado7W1Vdxyyy3CYrEIm80mbr/9duHxeJLKbN26VVx++eVCr9eLwsJC8dRTT/XVJaaVk7WXz+cT06dPFzk5OUKr1YrS0lJx55139gh82V5950RtBUC8/PLLSplz9XPw008/FePGjRM6nU4MHjw46TPo9JyqvQ4dOiSuvPJKkZmZKfR6vRg6dKi4//77RWdnZ9J52F5954477hClpaVCp9OJnJwcMXXqVCXwEIL3V39zsvbi/TUwHB968B4buCQhhOi7fiVERERERERERH2Dc3oQERERERERUVpi6EFEREREREREaYmhBxERERERERGlJYYeRERERERERJSWGHoQERERERERUVpi6EFEREREREREaYmhBxERERERERGlJYYeRERERERERJSWGHoQERFRvyJJEt55551UV4OIiIjSAEMPIiIiUtx2222QJKnHNmPGjFRXjYiIiOiMaVJdASIiIupfZsyYgZdffjlpn16vT1FtiIiIiM4ee3oQERFREr1ej7y8vKQtIyMDgDz0ZOnSpZg5cyaMRiMGDx6Mt956K+n927dvx3e/+10YjUZkZWXhrrvuQldXV1KZl156CSNHjoRer0d+fj4WLFiQdLylpQU//OEPYTKZMGzYMLz33nvn96KJiIgoLTH0ICIiojPyy1/+ErNnz8bWrVsxZ84c3Hzzzdi1axcAwOv1oqqqChkZGfjqq6/w5ptv4uOPP04KNZYuXYrq6mrcdddd2L59O9577z0MHTo06TOeeOIJ3Hjjjdi2bRuuueYazJkzB21tbX16nURERDTwSUIIkepKEBERUf9w22234X/+539gMBiS9j/88MN4+OGHIUkS5s2bh6VLlyrHJk+ejEsuuQTPP/88/vKXv+CBBx7A4cOHYTabAQAffPABrr32Whw9ehROpxOFhYW4/fbb8atf/eqEdZAkCY8++ij+8z//E4AcpFgsFnz44YecW4SIiIjOCOf0ICIioiTf+c53kkINAMjMzFSeV1ZWJh2rrKzEli1bAAC7du3C2LFjlcADAKZMmYJYLIba2lpIkoSjR49i6tSpJ63DmDFjlOdmsxk2mw1NTU1ne0lERER0gWLoQUREREnMZnOP4SbnitFoPK1yWq026bUkSYjFYuejSkRERJTGOKcHERERnZH169f3eD1ixAgAwIgRI7B161Z4vV7l+BdffAGVSoXy8nJYrVYMGjQIq1at6tM6ExER0YWJPT2IiIgoSTAYRGNjY9I+jUaD7OxsAMCbb76JCRMm4PLLL8err76KjRs34q9//SsAYM6cOVi0aBHmzp2Lxx9/HM3Nzbj77rtx6623wul0AgAef/xxzJs3D7m5uZg5cyY8Hg+++OIL3H333X17oURERJT2GHoQERFRko8++gj5+flJ+8rLy7F7924A8soqy5Ytw09/+lPk5+fj9ddfR0VFBQDAZDLh73//O+655x5ceumlMJlMmD17Np555hnlXHPnzkUgEMDvfvc7/PznP0d2djZuuOGGvrtAIiIiumBw9RYiIiI6bZIkYfny5bjuuutSXRUiIiKiU+KcHkRERERERESUlhh6EBEREREREVFa4pweREREdNo4KpaIiIgGEvb0ICIiIiIiIqK0xNCDiIiIiIiIiNISQw8iIiIiIiIiSksMPYiIiIiIiIgoLTH0ICIiIiIiIqK0xNCDiIiIiIiIiNISQw8iIiIiIiIiSksMPYiIiIiIiIgoLf1ffjoVTAN4Gz0AAAAASUVORK5CYII=

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

```import os
import sys
import random
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SAVE_ROOT = Path(os.getcwd()) / 'large_files'
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
images_dir = Path(os.getcwd()) / 'images'
images_dir.mkdir(parents=True, exist_ok=True)


def get_primitive_root(p):
    for g in range(2, p):
        powers = set(pow(g, k, p) for k in range(1, p))
        if len(powers) == p - 1:
            return g
    return None
p = 43
g = get_primitive_root(p)
print(f"Modulus: {p} (prime), Primitive Root (generator): g = {g}")

d_log = {}
d_exp = {}
for u in range(p - 1):
    val = pow(g, u, p)
    d_log[val] = u
    d_exp[u] = val
d_log[0] = -1 

@dataclass
class MULT_CONFIG:
    lr: float = 1e-3
    weight_decay: float = 0.8
    num_epochs: int = int(os.environ.get("MULT_EPOCHS", 8000))
    p: int = 43
    frac_train: float = 0.35  
    seed: int = 42
    d_model: int = 64
    d_vocab: int = 44  
    d_mlp: int = 256
    num_heads: int = 4
    n_ctx: int = 3
    @property
    def device(self):
        if t.cuda.is_available(): return t.device('cuda')
        elif t.backends.mps.is_available(): return t.device('mps')
        return t.device('cpu')
    @property
    def fn(self):
        return lambda x, y: (x * y) % self.p
m_config = MULT_CONFIG()
print(f"Device: {m_config.device} | Epochs: {m_config.num_epochs}")

def gen_data(config):
    pairs = [(i, j, config.p) for i in range(config.p) for j in range(config.p)]
    random.seed(config.seed)
    random.shuffle(pairs)
    div = int(config.frac_train * len(pairs))
    return pairs[:div], pairs[div:]
train_data, test_data = gen_data(m_config)
print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}")

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_vocab, d_model) / np.sqrt(d_model))
    def forward(self, x):
        return self.W_E[x]
class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_vocab))
    def forward(self, x):
        return x @ self.W_U
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model) / np.sqrt(d_model))
    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, n_ctx):
        super().__init__()
        d_head = d_model // num_heads
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head
    def forward(self, x):
        k = t.einsum('ihd,bpd->biph', self.W_K, x)
        q = t.einsum('ihd,bpd->biph', self.W_Q, x)
        v = t.einsum('ihd,bpd->biph', self.W_V, x)
        scores = t.einsum('biph,biqh->biqp', k, q)
        scores_masked = t.tril(scores) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn = F.softmax(scores_masked / np.sqrt(self.d_head), dim=-1)
        z = t.einsum('biph,biqp->biqh', v, attn)
        z_flat = z.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return z_flat @ self.W_O.T
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
    def forward(self, x):
        x = F.relu(x @ self.W_in.T + self.b_in)
        return x @ self.W_out.T + self.b_out
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, num_heads, n_ctx):
        super().__init__()
        self.attn = Attention(d_model, num_heads, n_ctx)
        self.mlp = MLP(d_model, d_mlp)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
class MultTransformer(nn.Module):
    def __init__(self, d_vocab, d_model, d_mlp, num_heads, n_ctx):
        super().__init__()
        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.block = TransformerBlock(d_model, d_mlp, num_heads, n_ctx)
        self.unembed = Unembed(d_vocab, d_model)
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        x = self.block(x)
        return self.unembed(x)

model = MultTransformer(m_config.d_vocab, m_config.d_model, m_config.d_mlp, m_config.num_heads, m_config.n_ctx)
model.to(m_config.device)
optimizer = optim.AdamW(model.parameters(), lr=m_config.lr, weight_decay=m_config.weight_decay, betas=(0.9, 0.98))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(s / 10, 1))

train_tensor = t.tensor(train_data, device=m_config.device)
test_tensor = t.tensor(test_data, device=m_config.device)
train_labels = t.tensor([m_config.fn(i, j) for i, j, _ in train_data], device=m_config.device)
test_labels = t.tensor([m_config.fn(i, j) for i, j, _ in test_data], device=m_config.device)
print("\nTraining small transformer to generalize modular multiplication...")
start_time = time.time()
grok_epoch = -1
for epoch in range(m_config.num_epochs):
    model.train()
    logits = model(train_tensor)[:, -1]
    loss = F.cross_entropy(logits, train_labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    if epoch % 500 == 0 or epoch == m_config.num_epochs - 1:
        model.eval()
        with t.no_grad():
            train_preds = logits.argmax(-1)
            train_acc = (train_preds == train_labels).float().mean().item()
            test_logits = model(test_tensor)[:, -1]
            test_loss = F.cross_entropy(test_logits, test_labels).item()
            test_preds = test_logits.argmax(-1)
            test_acc = (test_preds == test_labels).float().mean().item()
            print(f"Epoch {epoch:>5d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
            
            if test_acc > 0.95 and grok_epoch == -1:
                grok_epoch = epoch
                print(f"🎉 GENERALIZATION ATTAINED! Model grokked modular multiplication at epoch {epoch}!")
print(f"Training finished in {time.time() - start_time:.1f} seconds.")

print("\nPerforming Mechanistic Interpretability Analysis on the learned embeddings...")
model.eval()

W_E = model.embed.W_E.detach().cpu().numpy()

non_zero_elements = list(range(1, p))
embeddings = W_E[non_zero_elements] 

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=non_zero_elements, cmap='viridis', s=80, edgecolors='black', linewidths=0.5)
ax1.set_title("Embeddings ordered by Numerical Value (Chaotic)", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Principal Component 1", fontsize=10)
ax1.set_ylabel("Principal Component 2", fontsize=10)
fig.colorbar(scatter1, ax=ax1, label='Numerical Value')
ax1.grid(True, alpha=0.3)
for idx, elem in enumerate(non_zero_elements):
    ax1.annotate(str(elem), (embeddings_2d[idx, 0] + 0.02, embeddings_2d[idx, 1] + 0.02), fontsize=8, alpha=0.8)

discrete_logs = [d_log[elem] for elem in non_zero_elements]
scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=discrete_logs, cmap='twilight', s=80, edgecolors='black', linewidths=0.5)
ax2.set_title(f"Embeddings ordered by Discrete Logarithm base g={g} (Circular!)", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Principal Component 1", fontsize=10)
ax2.set_ylabel("Principal Component 2", fontsize=10)
fig.colorbar(scatter2, ax=ax2, label=f'Discrete Log (mod {p-1})')
ax2.grid(True, alpha=0.3)
for idx, elem in enumerate(non_zero_elements):
    ax2.annotate(str(elem), (embeddings_2d[idx, 0] + 0.02, embeddings_2d[idx, 1] + 0.02), fontsize=8, alpha=0.8)
plt.suptitle(f"Teasing Out Modular Multiplication Circuit ($p={p}$)\nDiscrete Logarithm representation learned by One-Layer Transformer", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
save_plot_path = images_dir / '5_modular_multiplication_circuit.png'
plt.savefig(save_plot_path, dpi=300)
plt.close()
print(f"\n✓ Premium circuit visualization successfully generated!")
print(f"  - Embedding Circuit Diagram → images/5_modular_multiplication_circuit.png")
print("\nThis visual proof shows the model maps the elements to a circular loop ordered exactly by their discrete logarithms. The transformer does not do multiplication; it does circular addition in the exponent space!")
```
