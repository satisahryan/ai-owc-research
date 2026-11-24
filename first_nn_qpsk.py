"""
Day 1 exercise: a tiny neural network learns to demodulate QPSK over AWGN.
Runs on CPU in a few seconds. Compare its symbol error rate (SER) with the
optimal minimum-distance detector.

Run:  conda activate ai
      python first_nn_qpsk.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

# QPSK constellation (unit energy)
const = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float32) / 2**0.5

def make_data(n, snr_db):
    labels = torch.randint(0, 4, (n,))
    x = const[labels]
    noise_std = (10 ** (-snr_db / 10) / 2) ** 0.5          # per real dimension
    y = x + noise_std * torch.randn_like(x)
    return y, labels

# Model: 2 inputs (I, Q) -> 4 classes
model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train at a mix of SNRs
for step in range(3000):
    y, labels = make_data(512, snr_db=float(torch.empty(1).uniform_(0, 12)))
    loss = loss_fn(model(y), labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# Evaluate: neural net vs optimal minimum-distance detector
print("\nSNR(dB)   SER (NN)    SER (min-distance)")
with torch.no_grad():
    for snr in range(0, 13, 2):
        y, labels = make_data(200_000, snr)
        ser_nn = (model(y).argmax(1) != labels).float().mean().item()
        ml = torch.cdist(y, const).argmin(1)
        ser_ml = (ml != labels).float().mean().item()
        print(f"{snr:5d}     {ser_nn:.5f}     {ser_ml:.5f}")
