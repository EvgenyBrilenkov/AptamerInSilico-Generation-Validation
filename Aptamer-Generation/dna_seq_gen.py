import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from metrics import compute_avg_levenshtein, compute_gc_mean, compute_mfe
from torch.utils.data import DataLoader, Dataset


# ------------------------
# Reproducibility
# ------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


# ------------------------
# Config
# ------------------------
SEQ_LEN = 80
VOCAB = ["A", "C", "G", "T", "X"]
V2I = {c: i for i, c in enumerate(VOCAB)}
I2V = {i: c for i, c in enumerate(VOCAB)}

LOG_FILE = "training_results/dna_generation_results.txt"
VOCAB_SIZE = len(VOCAB)
NOISE_DIM = 64
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Encoding / decoding
# ------------------------
def seq_to_onehot(seq):
    seq = list(seq[:SEQ_LEN])
    seq += ["X"] * (SEQ_LEN - len(seq))

    x = np.zeros((SEQ_LEN, VOCAB_SIZE), dtype=np.float32)
    for i, c in enumerate(seq):
        x[i, V2I[c]] = 1.0
    return x


def cut_at_X(seq):
    return seq.split("X")[0] if "X" in seq else seq


def decode(batch):
    idxs = torch.argmax(batch, dim=-1)
    out = []
    for row in idxs:
        s = "".join(I2V[i.item()] for i in row)
        out.append(cut_at_X(s))
    return out


# ------------------------
# Dataset
# ------------------------
class AptamerDataset(Dataset):
    def __init__(self, seqs):
        self.data = [seq_to_onehot(s) for s in seqs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor(self.data[i])


# ------------------------
# Models
# ------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=1)
        self.c2 = nn.Conv1d(c, c, 3, padding=1)
        self.n1 = nn.BatchNorm1d(c)
        self.n2 = nn.BatchNorm1d(c)

    def forward(self, x):
        r = x
        x = F.relu(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        return F.relu(x + r)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(NOISE_DIM, 128 * (SEQ_LEN // 4))

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.res = nn.Sequential(ResBlock(32), ResBlock(32))

        self.out = nn.Conv1d(32, VOCAB_SIZE, 1)

    def forward(self, z, tau=1.0, hard=False):
        x = self.fc(z)
        x = x.view(-1, 128, SEQ_LEN // 4)

        x = self.up(x)
        x = self.res(x)
        x = self.out(x)

        x = x.transpose(1, 2)
        return F.gumbel_softmax(x, tau=tau, hard=hard)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(VOCAB_SIZE, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Linear(256 * (SEQ_LEN // 4), 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(-1)


# ------------------------
# WGAN-GP
# ------------------------
def gradient_penalty(D, real, fake):
    a = torch.rand(real.size(0), 1, 1, device=DEVICE)
    inter = (a * real + (1 - a) * fake).requires_grad_(True)

    d = D(inter)

    g = torch.autograd.grad(
        outputs=d, inputs=inter, grad_outputs=torch.ones_like(d), create_graph=True
    )[0]

    g = g.reshape(g.size(0), -1)
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


# ------------------------
# k-mer stats
# ------------------------
def kmer(batch, k):
    B, L, V = batch.shape
    outs = []

    for i in range(L - k + 1):
        w = batch[:, i : i + k, :].reshape(B, -1)
        outs.append(w)

    return torch.stack(outs, dim=1).mean(dim=1)


# ------------------------
# entropy regularization
# ------------------------
def entropy(x):
    p = x.mean(dim=1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()


# ------------------------
# metrics
# ------------------------
def compute_metrics(real, fake):
    gc = compute_gc_mean(fake)
    mfe = np.nanmean(compute_mfe(fake))
    lev = compute_avg_levenshtein(real, fake)
    length = np.mean([len(s) for s in fake])
    return gc, mfe, lev, length


def compute_real_metrics(real_seqs):
    gc = compute_gc_mean(real_seqs)
    mfe = np.nanmean(compute_mfe(real_seqs))
    length = np.mean([len(s) for s in real_seqs])
    return gc, mfe, length


# ------------------------
# Training step
# ------------------------
def train_step(G, D, loader, opt_g, opt_d, tau):

    G.train()
    D.train()

    for real in loader:
        real = real.to(DEVICE)

        # --------------------
        # Critic
        # --------------------
        for _ in range(3):
            z = torch.randn(real.size(0), NOISE_DIM, device=DEVICE)

            fake = G(z, tau=tau, hard=False).detach()

            loss_d = D(fake).mean() - D(real).mean()
            loss_d += 10 * gradient_penalty(D, real, fake)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        # --------------------
        # Generator
        # --------------------
        z = torch.randn(real.size(0), NOISE_DIM, device=DEVICE)
        fake = G(z, tau=tau, hard=False)

        adv = -D(fake).mean()

        k2 = F.mse_loss(kmer(fake, 2), kmer(real, 2).detach())
        k3 = F.mse_loss(kmer(fake, 3), kmer(real, 3).detach())

        ent = entropy(fake)

        loss_g = adv + 2 * k2 + 5 * k3 - 0.05 * ent

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()


def log_epoch(f, text):
    f.write(text + "\n")
    f.flush()


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    df = pd.read_csv("train_dna_dataset.csv")
    df = df.drop_duplicates(subset=["apt_seq"], keep="first")
    df = df[~df["apt_seq"].str.contains("U", na=False)]
    seqs = df["apt_seq"].astype(str).tolist()

    dataset = AptamerDataset(seqs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Critic().to(DEVICE)

    opt_g = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    opt_d = optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.9))

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== TRAINING START ===\n")

        real_clean = [cut_at_X(s) for s in seqs]
        gc_r, mfe_r, len_r = compute_real_metrics(real_clean)

        f.write("[REAL DATA]\n")
        f.write(f"GC: {gc_r:.3f} | MFE: {mfe_r:.2f} | Len: {len_r:.2f}\n\n")

        print("Training...")

        for epoch in range(200):
            tau = max(0.6, 1.0 - epoch * 0.002)

            train_step(G, D, loader, opt_g, opt_d, tau)

            # --------------------
            # EVAL
            # --------------------
            G.eval()
            with torch.no_grad():
                z = torch.randn(100, NOISE_DIM, device=DEVICE)
                fake = G(z, tau=tau, hard=True)

                fake_seqs = decode(fake)

                gc_f, mfe_f, lev_f, len_f = compute_metrics(real_clean, fake_seqs)

                d_loss = None
                g_loss = None

                # лог строка
                header = f"\n===== EPOCH {epoch} =====\n"
                metrics = (
                    f"[FAKE]\n"
                    f"GC: {gc_f:.3f} | MFE: {mfe_f:.2f} | Lev: {lev_f:.2f} | Len: {len_f:.2f}\n\n"
                    f"[REAL]\n"
                    f"GC: {gc_r:.3f} | MFE: {mfe_r:.2f} | Len: {len_r:.2f}\n\n"
                )

                samples = "[SAMPLES]\n" + "\n".join(random.sample(fake_seqs, 5)) + "\n"

                full_log = header + metrics + samples + ("-" * 60 + "\n")

                # print
                print(full_log)

                # write file
                log_epoch(f, full_log)

    torch.save(G.state_dict(), "weights/generator_dna.pth")
    torch.save(D.state_dict(), "weights/critic_dna.pth")
    print("Models saved.")
