"""EDA for aptamer generation and binding-screening tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    both_labels = (
        pairs.groupby(["apt_seq", "target_seq"])["label"].nunique().ge(2).sum()
    )
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
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
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
    candidates, candidate_info = load_candidates()
    _, docking_info = load_docking()

    summary = {
        "dna": pair_stats(dna),
        "rna": pair_stats(rna),
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
