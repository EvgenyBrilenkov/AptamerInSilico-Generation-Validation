import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance
import RNA  # from ViennaRNA

from data_preparing import VOCAB, train_seqs


def decode_onehot_batch(onehot_batch, vocab=VOCAB):
    """Преобразует батч one-hot в список строк без PAD."""
    B, L, V = onehot_batch.shape
    probs = onehot_batch.view(-1, V)
    idxs = torch.multinomial(probs, 1).view(B, L)
    seqs = []
    for row in idxs:
        seq = ''.join(vocab[idx] for idx in row)
        seq = seq.split('X')  # убираем паддинги
        seqs.append(seq[0])
    return seqs

def decode_onehot_batch_visual(onehot_batch, vocab=VOCAB):
    """Преобразует батч one-hot в список строк без PAD."""
    batch_size, seq_len, vocab_size = onehot_batch.shape
    idxs = torch.argmax(onehot_batch, dim=-1)
    seqs = []
    for i in range(batch_size):
        chars = [vocab[idx] for idx in idxs[i]]
        seq = ''.join(c for c in chars)
        seqs.append(seq)
    return seqs


def gc_content(seq):
    if len(seq) == 0:
        return 0.0
    return (seq.count('G') + seq.count('C')) / len(seq)


def compute_gc_mean(seqs):
    return np.mean([gc_content(s) for s in seqs if len(s) > 0])


def compute_lengths(seqs):
    return [len(s) for s in seqs]


def compute_mfe(seqs):
    """Возвращает список MFE значений (в ккал/моль)."""
    mfe_vals = []
    for seq in seqs:
        if len(seq) == 0:
            continue
        try:
            _, mfe = RNA.fold(seq)
            mfe_vals.append(mfe)
        except Exception as e:
            print(f"ViennaRNA error on '{seq}': {e}")
            mfe_vals.append(np.nan)
    return mfe_vals


def levenshtein_to_closest(real_seqs, fake_seq):
    """Минимальное расстояние Левенштейна от fake_seq до любого real_seq."""
    return min(levenshtein_distance(fake_seq, r) for r in real_seqs if len(r) > 0)


def compute_avg_levenshtein(real_seqs, fake_seqs):
    dists = []
    for f in fake_seqs:
        if len(f) == 0:
            continue
        d = levenshtein_to_closest(real_seqs, f)
        dists.append(d)
    return np.mean(dists) if dists else float('nan')