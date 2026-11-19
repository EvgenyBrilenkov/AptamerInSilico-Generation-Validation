"""EDA for aptamer generation and binding-screening tables."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Aptamer-Generation" / "data"
CANDIDATES = ROOT / "Aptamer-Validation" / "docking_candidates" / "candidates_for_docking.csv"
DOCKING = ROOT / "Aptamer-Validation" / "docking_results" / "final_ranked_candidates_hdock.csv"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DNA_ALPHABET = set("ACGT")
RNA_ALPHABET = set("ACGU")
AA = set("ACDEFGHIKLMNPQRSTVWY")


def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return float("nan")
    return (seq.count("G") + seq.count("C")) / len(seq)


def load_split(path: Path, kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["kind"] = kind
    frame["apt_seq"] = frame["apt_seq"].fillna("").astype(str).str.strip()
    frame["target_seq"] = frame["target_seq"].fillna("").astype(str).str.strip()
    frame["label"] = frame["label"].astype(int)
    frame["apt_len"] = frame["apt_seq"].str.len()
    frame["target_len"] = frame["target_seq"].str.len()
    frame["gc"] = frame["apt_seq"].map(gc_content)
    return frame


def alphabet_issues(series: pd.Series, allowed: set[str]) -> dict:
    bad_rows = 0
    bad_chars: set[str] = set()
    for seq in series:
        extra = set(seq.upper()) - allowed
        if extra:
            bad_rows += 1
            bad_chars |= extra
    return {"rows_with_unexpected_chars": bad_rows, "unexpected_chars": sorted(bad_chars)}


def pair_stats(frame: pd.DataFrame) -> dict:
    pairs = frame[["apt_seq", "target_seq", "label"]]
    duplicated_pairs = int(pairs.duplicated().sum())
    both_labels = pairs.groupby(["apt_seq", "target_seq"])["label"].nunique().ge(2).sum()
    pos = pairs[pairs["label"] == 1]
    neg = pairs[pairs["label"] == 0]
    pos_set = set(zip(pos["apt_seq"], pos["target_seq"]))
    pos_apts = set(pos["apt_seq"])
    pos_targets = set(pos["target_seq"])
    neg_outside = int(
        ((~neg["apt_seq"].isin(pos_apts)) | (~neg["target_seq"].isin(pos_targets))).sum()
    )
    return {
        "rows": int(len(frame)),
        "missing_apt": int((frame["apt_seq"] == "").sum()),
        "missing_target": int((frame["target_seq"] == "").sum()),
        "unique_aptamers": int(frame["apt_seq"].nunique()),
        "unique_targets": int(frame["target_seq"].nunique()),
        "unique_pairs": int(pairs.drop_duplicates().shape[0]),
        "duplicated_rows": duplicated_pairs,
        "pairs_with_both_labels": int(both_labels),
        "label_counts": {
            str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()
        },
        "positive_pairs": int(len(pos_set)),
        "negative_rows_using_unseen_apt_or_target": neg_outside,
        "apt_len": frame["apt_len"].describe().round(2).to_dict(),
        "target_len": frame["target_len"].describe().round(2).to_dict(),
        "gc": frame["gc"].describe().round(3).to_dict(),
        "apt_longer_than_80": int((frame["apt_len"] > 80).sum()),
        "apt_shorter_than_10": int((frame["apt_len"] < 10).sum()),
        "target_longer_than_1000": int((frame["target_len"] > 1000).sum()),
        "gc_by_label": {
            str(k): round(float(v), 3) for k, v in frame.groupby("label")["gc"].mean().items()
        },
        "apt_len_by_label": {
            str(k): round(float(v), 2) for k, v in frame.groupby("label")["apt_len"].mean().items()
        },
    }


def save_length_gc(dna: pd.DataFrame, rna: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, frame, title in (
        (axes[0], dna, "DNA"),
        (axes[1], rna, "RNA"),
    ):
        for label, color in ((0, "#4C78A8"), (1, "#F58518")):
            part = frame.loc[frame["label"] == label, "apt_len"]
            ax.hist(part, bins=30, alpha=0.65, label=f"label={label}", color=color)
        ax.axvline(80, color="black", linestyle="--", linewidth=1, label="длина генератора 80")
        ax.set_title(f"Длина аптамера, {title}")
        ax.set_xlabel("нуклеотиды")
        ax.set_ylabel("число пар")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "aptamer_length.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    combined = pd.concat([dna, rna], ignore_index=True)
    means = combined.groupby(["kind", "label"])["gc"].mean().unstack()
    means.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"], rot=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("средний GC")
    ax.set_xlabel("")
    ax.set_title("GC-состав аптамеров по типу и метке")
    ax.legend(title="label")
    fig.tight_layout()
    fig.savefig(FIG / "gc_by_label.png", dpi=140)
    plt.close(fig)


def save_balance_and_proteins(dna: pd.DataFrame, rna: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = pd.DataFrame(
        {
            "DNA": dna["label"].value_counts().sort_index(),
            "RNA": rna["label"].value_counts().sort_index(),
        }
    )
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"], rot=0)
    ax.set_xlabel("метка связывания")
    ax.set_ylabel("число пар")
    ax.set_title("Баланс меток в обучающих таблицах")
    fig.tight_layout()
    fig.savefig(FIG / "label_balance.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dna["target_len"], bins=30, alpha=0.65, label="DNA-пары", color="#4C78A8")
    ax.hist(rna["target_len"], bins=30, alpha=0.65, label="RNA-пары", color="#F58518")
    ax.axvline(1000, color="black", linestyle="--", linewidth=1, label="1000 аминокислот")
    ax.set_xlabel("длина белка")
    ax.set_ylabel("число пар")
    ax.set_title("Длина белковой мишени")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "protein_length.png", dpi=140)
    plt.close(fig)


def load_candidates() -> tuple[pd.DataFrame, dict]:
    raw = CANDIDATES.read_text(encoding="utf-8").splitlines()
    frame = pd.read_csv(CANDIDATES, skiprows=1)
    info = {
        "file_lines": len(raw),
        "junk_first_line": raw[0] if raw else "",
        "rows": int(len(frame)),
        "unique_aptamers": int(frame["apt_seq"].nunique()),
        "unique_targets": int(frame["target_seq"].nunique()),
        "unique_target_names": sorted(frame["target_name"].dropna().unique().tolist()),
        "true_label_counts": {
            str(k): int(v) for k, v in frame["true_label"].value_counts().sort_index().items()
        },
        "proba_min": {
            col: round(float(frame[col].min()), 4)
            for col in ("proba_1_apipred", "proba_1_aptatrans", "proba_1_aptanet")
        },
    }
    return frame, info


def roc_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    paired = pd.DataFrame({"y": labels, "s": scores}).dropna()
    y = paired["y"].to_numpy(dtype=int)
    score = paired["s"].to_numpy(dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    y_sorted = y[order]
    score_sorted = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        stop = start
        while stop + 1 < len(score) and score_sorted[stop + 1] == score_sorted[start]:
            stop += 1
        ranks[start : stop + 1] = (start + stop) / 2 + 1
        start = stop + 1
    sum_pos = float(ranks[y_sorted == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return round(auc, 3)


def feature_separation(frame: pd.DataFrame) -> dict:
    hard_targets = set(frame.groupby("target_seq")["label"].nunique().loc[lambda s: s.eq(2)].index)
    hard = frame[frame["target_seq"].isin(hard_targets)]
    apt = frame.groupby("apt_seq", as_index=False).agg(
        label=("label", "max"), apt_len=("apt_len", "first"), gc=("gc", "first")
    )
    return {
        "rows_auc_apt_len": roc_auc(frame["label"], frame["apt_len"]),
        "rows_auc_gc": roc_auc(frame["label"], frame["gc"]),
        "hard_rows": int(len(hard)),
        "hard_targets": int(len(hard_targets)),
        "hard_positive_rate": round(float(hard["label"].mean()), 3) if len(hard) else None,
        "hard_auc_apt_len": roc_auc(hard["label"], hard["apt_len"]) if len(hard) else None,
        "hard_auc_gc": roc_auc(hard["label"], hard["gc"]) if len(hard) else None,
        "unique_apt_auc_len": roc_auc(apt["label"], apt["apt_len"]),
        "unique_apt_auc_gc": roc_auc(apt["label"], apt["gc"]),
    }


def protein_only_baseline(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby("target_seq")["label"]
    total = grouped.transform("sum")
    count = grouped.transform("size")
    loo = (total - frame["label"]) / (count - 1).replace(0, np.nan)
    loo = loo.fillna(float(frame["label"].mean()))
    hard_targets = set(frame.groupby("target_seq")["label"].nunique().loc[lambda s: s.eq(2)].index)
    hard = frame["target_seq"].isin(hard_targets)
    return {
        "auc_protein_loo_rate": roc_auc(frame["label"], loo),
        "hard_auc_protein_loo_rate": roc_auc(frame.loc[hard, "label"], loo[hard]),
    }


def negative_ease(frame: pd.DataFrame) -> dict:
    pos_targets = set(frame.loc[frame["label"] == 1, "target_seq"])
    pos = frame[frame["label"] == 1]
    neg = frame[frame["label"] == 0]
    easy = neg[~neg["target_seq"].isin(pos_targets)]
    hard = neg[neg["target_seq"].isin(pos_targets)]
    never_apt = set(frame.groupby("apt_seq")["label"].max().loc[lambda s: s.eq(0)].index)

    def block(part: pd.DataFrame) -> dict:
        return {
            "rows": int(len(part)),
            "mean_apt_len": round(float(part["apt_len"].mean()), 2) if len(part) else None,
            "mean_gc": round(float(part["gc"].mean()), 3) if len(part) else None,
        }

    return {
        "positives": block(pos),
        "negatives_on_positive_targets": block(hard),
        "negatives_on_targets_without_positives": block(easy),
        "negative_rows_from_never_positive_aptamers": int(neg["apt_seq"].isin(never_apt).sum()),
    }


def aptamer_pools(frame: pd.DataFrame) -> dict:
    apt = frame.groupby("apt_seq", as_index=False).agg(
        ever_positive=("label", "max"),
        apt_len=("apt_len", "first"),
        gc=("gc", "first"),
        rows=("label", "size"),
    )
    never = apt[apt["ever_positive"] == 0]
    ever = apt[apt["ever_positive"] == 1]
    return {
        "unique": int(len(apt)),
        "ever_positive": {
            "n": int(len(ever)),
            "mean_len": round(float(ever["apt_len"].mean()), 2),
            "median_len": round(float(ever["apt_len"].median()), 2),
            "mean_gc": round(float(ever["gc"].mean()), 3),
            "share_of_generator": round(len(ever) / len(apt), 3),
        },
        "never_positive": {
            "n": int(len(never)),
            "mean_len": round(float(never["apt_len"].mean()), 2),
            "median_len": round(float(never["apt_len"].median()), 2),
            "mean_gc": round(float(never["gc"].mean()), 3),
            "share_of_generator": round(len(never) / len(apt), 3),
            "rows": int(never["rows"].sum()),
        },
    }


def _kmer_matrix(seqs: list[str], k: int = 3) -> np.ndarray:
    prepared = [seq.upper() for seq in seqs]
    vocab = sorted({seq[i : i + k] for seq in prepared for i in range(max(0, len(seq) - k + 1))})
    index = {token: i for i, token in enumerate(vocab)}
    matrix = np.zeros((len(prepared), max(len(vocab), 1)), dtype=float)
    for row, seq in enumerate(prepared):
        for i in range(max(0, len(seq) - k + 1)):
            matrix[row, index[seq[i : i + k]]] += 1
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def _pool_cosine(left: list[str], right: list[str], k: int = 3) -> float | None:
    if not left or not right:
        return None
    vocab: dict[str, int] = {}

    def add(seq: str, into: np.ndarray) -> None:
        seq = seq.upper()
        for i in range(max(0, len(seq) - k + 1)):
            token = seq[i : i + k]
            if token not in vocab:
                vocab[token] = len(vocab)
            column = vocab[token]
            if column >= into.shape[0]:
                into.resize(column + 1, refcheck=False)
            into[column] += 1

    left_vec = np.zeros(1, dtype=float)
    right_vec = np.zeros(1, dtype=float)
    for seq in left:
        add(seq, left_vec)
    for seq in right:
        add(seq, right_vec)
    width = max(len(vocab), 1)
    if left_vec.shape[0] < width:
        left_vec.resize(width, refcheck=False)
    if right_vec.shape[0] < width:
        right_vec.resize(width, refcheck=False)
    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    if denom == 0:
        return None
    return round(float(left_vec @ right_vec / denom), 3)


def sequence_redundancy(frame: pd.DataFrame, threshold: float = 0.9) -> dict:
    apt = frame.groupby("apt_seq", as_index=False).agg(ever_positive=("label", "max"))
    seqs = apt["apt_seq"].tolist()
    normed = _kmer_matrix(seqs, k=3)
    similarity = normed @ normed.T
    np.fill_diagonal(similarity, -1)
    nearest = similarity.max(axis=1)
    n = len(seqs)
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    rows, cols = np.triu_indices(n, 1)
    close = np.flatnonzero(similarity[rows, cols] >= threshold)
    for idx in close:
        left_root = find(int(rows[idx]))
        right_root = find(int(cols[idx]))
        if left_root != right_root:
            parent[right_root] = left_root
    sizes = Counter(find(i) for i in range(n))
    clustered = [size for size in sizes.values() if size >= 2]
    never = apt["ever_positive"].eq(0).to_numpy()
    ever = ~never
    pool = _pool_cosine(apt.loc[ever, "apt_seq"].tolist(), apt.loc[never, "apt_seq"].tolist())
    return {
        "kmer": 3,
        "cosine_threshold": threshold,
        "unique_aptamers": n,
        "median_nearest_cosine": round(float(np.median(nearest)), 3),
        "pairs_at_or_above_threshold": int(len(close)),
        "clusters_of_at_least_2": len(clustered),
        "largest_cluster": int(max(sizes.values()) if sizes else 0),
        "aptamers_in_those_clusters": int(sum(clustered)),
        "kmer_cosine_ever_vs_never_positive": pool,
    }


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def random_sequence_baseline(frame: pd.DataFrame, rna: bool, seed: int = 42) -> dict:
    seqs = frame.drop_duplicates("apt_seq")["apt_seq"].str.upper()
    # Тот же отбор, что в dna_seq_gen.py / rna_seq_gen.py: чужая буква выкидывается.
    forbidden = "T" if rna else "U"
    seqs = seqs[~seqs.str.contains(forbidden, na=False)].tolist()
    at_pair = ("A", "U") if rna else ("A", "T")
    rng = np.random.default_rng(seed)
    random_seqs = []
    for seq in seqs:
        gc_count = seq.count("G") + seq.count("C")
        tokens = np.array(["S"] * gc_count + ["W"] * (len(seq) - gc_count))
        rng.shuffle(tokens)
        built = []
        for token in tokens:
            if token == "S":
                built.append(str(rng.choice(["G", "C"])))
            else:
                built.append(str(rng.choice(list(at_pair))))
        random_seqs.append("".join(built))
    by_length: dict[int, list[str]] = {}
    for seq in seqs:
        by_length.setdefault(len(seq), []).append(seq)
    distances = []
    for random_seq in random_seqs:
        best = len(random_seq)
        for delta in range(best + 1):
            for length in {len(random_seq) - delta, len(random_seq) + delta}:
                for real in by_length.get(length, []):
                    best = min(best, _levenshtein(random_seq, real))
            if delta >= best:
                break
        distances.append(best)
    random_gc = [gc_content(seq) for seq in random_seqs]
    real_gc = [gc_content(seq) for seq in seqs]
    return {
        "n": len(seqs),
        "real_mean_gc": round(float(np.mean(real_gc)), 3),
        "random_mean_gc": round(float(np.mean(random_gc)), 3),
        "real_mean_len": round(float(np.mean([len(seq) for seq in seqs])), 2),
        "random_mean_len": round(float(np.mean([len(seq) for seq in random_seqs])), 2),
        "random_mean_min_levenshtein": round(float(np.mean(distances)), 2),
        "random_median_min_levenshtein": round(float(np.median(distances)), 2),
        "note": "MFE не считался: в этом прогоне ViennaRNA не вызывался",
    }


def shortlist_novelty(candidates: pd.DataFrame, dna: pd.DataFrame, rna: pd.DataFrame) -> dict:
    rna_apt = set(rna["apt_seq"])
    dna_apt = set(dna["apt_seq"])
    rna_target = set(rna["target_seq"])
    dna_target = set(dna["target_seq"])
    buckets = []
    for row in candidates.itertuples(index=False):
        apt = str(row.apt_seq)
        target = str(row.target_seq)
        apt_seen = apt in rna_apt or apt.replace("U", "T") in dna_apt
        target_seen = target in rna_target or target in dna_target
        if apt_seen and target_seen:
            buckets.append("both_seen")
        elif apt_seen:
            buckets.append("new_target")
        elif target_seen:
            buckets.append("new_aptamer")
        else:
            buckets.append("both_new")
    labeled = candidates.copy()
    labeled["bucket"] = buckets
    counts = {
        name: int(labeled["bucket"].eq(name).sum())
        for name in (
            "both_seen",
            "new_target",
            "new_aptamer",
            "both_new",
        )
    }
    by_label = {}
    for bucket, part in labeled.groupby("bucket"):
        by_label[str(bucket)] = {
            str(k): int(v) for k, v in part["true_label"].value_counts().sort_index().items()
        }
    return {"rows": counts, "true_label_by_bucket": by_label}


def top5_on_shortlist(candidates: pd.DataFrame, docking: pd.DataFrame) -> dict:
    best = docking.groupby("candidate")["docking_score"].min().rename("best_docking")
    merged = candidates.merge(best, left_on="candidate_id", right_index=True, how="left")
    named = merged.dropna(subset=["target_name"])
    overlaps = []
    per_target = []
    for name, part in named.groupby("target_name"):
        by_model = part.sort_values("proba_1_aptanet", ascending=False).head(5)
        docked = part.dropna(subset=["best_docking"])
        by_dock = docked.sort_values("best_docking", ascending=True).head(5)
        model_ids = set(by_model["candidate_id"])
        dock_ids = set(by_dock["candidate_id"])
        union = model_ids | dock_ids
        overlap = len(model_ids & dock_ids) / len(union) if union and dock_ids else None
        overlaps.append(overlap)
        per_target.append(
            {
                "target_name": str(name),
                "candidates": int(len(part)),
                "true_positives": int((part["true_label"] == 1).sum()),
                "with_docking": int(docked["candidate_id"].nunique()),
                "top5_jaccard_aptanet_vs_docking": None if overlap is None else round(overlap, 3),
            }
        )
    usable = [value for value in overlaps if value is not None]
    return {
        "named_targets": int(named["target_name"].nunique()),
        "rows_without_target_name": int(merged["target_name"].isna().sum()),
        "targets_with_more_than_5": int((named.groupby("target_name").size() > 5).sum()),
        "mean_jaccard_top5": round(float(np.mean(usable)), 3) if usable else None,
        "per_target": per_target,
        "why_not_precision_at_5": (
            "В файле почти все true_label равны 1, а пары, которые консенсус отбросил, сюда не попали"
        ),
    }


def save_pool_lengths(dna: pd.DataFrame, rna: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, frame, title in ((axes[0], dna, "DNA"), (axes[1], rna, "RNA")):
        apt = frame.groupby("apt_seq", as_index=False).agg(
            ever_positive=("label", "max"), apt_len=("apt_len", "first")
        )
        for flag, color, label in (
            (1, "#F58518", "был позитив"),
            (0, "#4C78A8", "позитива не было"),
        ):
            part = apt.loc[apt["ever_positive"] == flag, "apt_len"]
            ax.hist(part, bins=25, alpha=0.7, color=color, label=f"{label}, n={len(part)}")
        ax.set_title(f"Длина уникальных аптамеров, {title}")
        ax.set_xlabel("нуклеотиды")
        ax.set_ylabel("число аптамеров")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "never_positive_length.png", dpi=140)
    plt.close(fig)


def load_docking() -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(DOCKING)
    best = frame.groupby("candidate")["docking_score"].min()
    info = {
        "rows": int(len(frame)),
        "unique_candidates": int(frame["candidate"].nunique()),
        "models_per_candidate": frame.groupby("candidate").size().describe().round(2).to_dict(),
        "score_min": round(float(frame["docking_score"].min()), 2),
        "score_max": round(float(frame["docking_score"].max()), 2),
        "score_median": round(float(frame["docking_score"].median()), 2),
        "best_per_candidate_median": round(float(best.median()), 2),
        "candidates_best_le_minus_1000": int((best <= -1000).sum()),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(frame["docking_score"], bins=25, color="#54A24B")
    ax.set_title("Docking score HDOCK по конформерам")
    ax.set_xlabel("docking score")
    ax.set_ylabel("число моделей")
    fig.tight_layout()
    fig.savefig(FIG / "docking_scores.png", dpi=140)
    plt.close(fig)
    return frame, info


def main() -> None:
    dna = load_split(DATA / "train_dna_dataset.csv", "DNA")
    rna = load_split(DATA / "train_rna_dataset.csv", "RNA")
    save_length_gc(dna, rna)
    save_balance_and_proteins(dna, rna)
    save_pool_lengths(dna, rna)
    candidates, candidate_info = load_candidates()
    docking, docking_info = load_docking()

    summary = {
        "dna": pair_stats(dna),
        "rna": pair_stats(rna),
        "separation": {"dna": feature_separation(dna), "rna": feature_separation(rna)},
        "protein_only": {"dna": protein_only_baseline(dna), "rna": protein_only_baseline(rna)},
        "negative_ease": {"dna": negative_ease(dna), "rna": negative_ease(rna)},
        "aptamer_pools": {"dna": aptamer_pools(dna), "rna": aptamer_pools(rna)},
        "generator_pools": {
            "dna": {
                **aptamer_pools(dna.loc[~dna["apt_seq"].str.contains("U")]),
                "dropped_with_U": int(
                    dna.loc[dna["apt_seq"].str.contains("U"), "apt_seq"].nunique()
                ),
            },
            "rna": {
                **aptamer_pools(rna.loc[~rna["apt_seq"].str.contains("T")]),
                "dropped_with_T": int(
                    rna.loc[rna["apt_seq"].str.contains("T"), "apt_seq"].nunique()
                ),
            },
        },
        "redundancy": {"dna": sequence_redundancy(dna), "rna": sequence_redundancy(rna)},
        "random_baseline": {
            "dna": random_sequence_baseline(dna, rna=False),
            "rna": random_sequence_baseline(rna, rna=True),
        },
        "shortlist_novelty": shortlist_novelty(candidates, dna, rna),
        "top5_shortlist": top5_on_shortlist(candidates, docking),
        "dna_alphabet": alphabet_issues(dna["apt_seq"], DNA_ALPHABET),
        "rna_alphabet": alphabet_issues(rna["apt_seq"], RNA_ALPHABET),
        "dna_target_alphabet": alphabet_issues(dna["target_seq"], AA),
        "rna_target_alphabet": alphabet_issues(rna["target_seq"], AA),
        "dna_contains_U": int(dna["apt_seq"].str.contains("U").sum()),
        "rna_contains_T": int(rna["apt_seq"].str.contains("T").sum()),
        "candidates": candidate_info,
        "docking": docking_info,
        "candidate_aptamers_in_dna_train": int(
            candidates["apt_seq"].str.replace("U", "T").isin(set(dna["apt_seq"])).sum()
        ),
        "candidate_aptamers_in_rna_train": int(
            candidates["apt_seq"].isin(set(rna["apt_seq"])).sum()
        ),
    }
    out = Path(__file__).resolve().parent / "eda_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
