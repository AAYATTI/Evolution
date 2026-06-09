#!/usr/bin/env python3
"""
===============================================================================
CODON USAGE BIAS COMPARATIVE PIPELINE (HepI vs OpsX)
===============================================================================

Author  : Aayatti Mallick Gupta
Purpose : Integrated analysis of codon usage bias metrics in coexisting
          Heptosyltransferase I variants:
          - HepI (rfaC / waaC)
          - OpsX

Metrics :
    1. GC3 content
    2. Effective Number of Codons (ENC)
    3. Relative Synonymous Codon Usage (RSCU)
    4. Codon Adaptation Index (CAI)

Outputs :
    - Summary statistics tables
    - Statistical test reports
    - Publication-quality plots

===============================================================================
"""

import os
import math
import argparse
import logging
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Bio import SeqIO
from scipy.stats import (
    shapiro,
    ttest_ind,
    mannwhitneyu,
    ttest_rel,
    wilcoxon,
    levene
)

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

OUTDIR = "codon_bias_results"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# CODON TABLE
# =============================================================================

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D',
    'GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGG':'W',
    'CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGA':'R','AGG':'R',
    'AGT':'S','AGC':'S','GGT':'G','GGC':'G','GGA':'G','GGG':'G'
}

AA_TABLE = defaultdict(list)
for c, aa in CODON_TABLE.items():
    AA_TABLE[aa].append(c)

# =============================================================================
# FASTA UTILITIES
# =============================================================================

def load_sequences(fasta_file):
    return [str(r.seq) for r in SeqIO.parse(fasta_file, "fasta")]

# =============================================================================
# GC3
# =============================================================================

def gc3(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
    third = [c[2] for c in codons if len(c) == 3]
    if not third:
        return None
    return sum(1 for n in third if n in "GC") / len(third)


def compute_gc3(fasta):
    return [gc3(s.upper()) for s in load_sequences(fasta) if gc3(s.upper()) is not None]

# =============================================================================
# ENC
# =============================================================================

def compute_enc(seq):
    seq = seq.upper()
    codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]

    F_values = []

    for aa, codon_list in AA_TABLE.items():
        if len(codon_list) == 1:
            continue

        counts = np.array([codons.count(c) for c in codon_list])
        n = counts.sum()

        if n <= 1:
            continue

        freq_sq = np.sum((counts / n) ** 2)
        F = (n * freq_sq - 1) / (n - 1)

        if F > 0:
            F_values.append(F)

    if not F_values:
        return None

    return 2 + sum(1 / f for f in F_values if f > 0)


def compute_enc(fasta):
    return [compute_enc(s) for s in load_sequences(fasta) if compute_enc(s)]

# =============================================================================
# RSCU
# =============================================================================

def rscu(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
    counts = defaultdict(int)

    for c in codons:
        counts[c] += 1

    rscu_vals = {}
    for aa, codon_list in AA_TABLE.items():
        total = sum(counts[c] for c in codon_list)
        if total == 0:
            continue
        for c in codon_list:
            rscu_vals[c] = (counts[c] * len(codon_list)) / total

    return rscu_vals


def compute_rscu_matrix(fasta):
    data = [rscu(str(r.seq).upper()) for r in SeqIO.parse(fasta, "fasta")]
    df = pd.DataFrame(data).fillna(0)
    return df

# =============================================================================
# CAI (simplified codon-weight model)
# =============================================================================

def compute_weights(fasta):
    counts = defaultdict(int)

    for seq in load_sequences(fasta):
        for i in range(0, len(seq)-2, 3):
            codon = seq[i:i+3]
            if codon in CODON_TABLE:
                counts[codon] += 1

    weights = {}
    for aa, codons in AA_TABLE.items():
        max_c = max([counts[c] for c in codons] + [1])
        for c in codons:
            weights[c] = counts[c] / max_c

    return weights


def cai(seq, weights):
    vals = []
    for i in range(0, len(seq)-2, 3):
        c = seq[i:i+3]
        if c in weights and weights[c] > 0:
            vals.append(math.log(weights[c]))

    return np.exp(np.mean(vals)) if vals else None


def compute_cai(fasta):
    weights = compute_weights(fasta)
    return [
        cai(str(r.seq).upper(), weights)
        for r in SeqIO.parse(fasta, "fasta")
    ]

# =============================================================================
# STATISTICS
# =============================================================================

def stats(a, b):
    normal = shapiro(a).pvalue > 0.05 and shapiro(b).pvalue > 0.05

    if normal:
        test = ttest_ind(a, b)
        name = "t-test"
    else:
        test = mannwhitneyu(a, b)
        name = "Mann-Whitney"

    return name, test

# =============================================================================
# PIPELINE RUNNER
# =============================================================================

def run(metric, hepi, opsx):

    logging.info(f"Running {metric} analysis")

    if metric == "GC3":
        a, b = compute_gc3(hepi), compute_gc3(opsx)

    elif metric == "ENC":
        a, b = compute_enc(hepi), compute_enc(opsx)

    elif metric == "CAI":
        a, b = compute_cai(hepi), compute_cai(opsx)

    elif metric == "RSCU":
        hepi_df = compute_rscu_matrix(hepi)
        opsx_df = compute_rscu_matrix(opsx)

        hepi_df.to_csv(f"{OUTDIR}/HepI_RSCU.tsv", sep="\t")
        opsx_df.to_csv(f"{OUTDIR}/OpsX_RSCU.tsv", sep="\t")
        return

    df = pd.DataFrame({
        "Gene": ["HepI"] * len(a) + ["OpsX"] * len(b),
        metric: list(a) + list(b)
    })

    df.to_csv(f"{OUTDIR}/{metric}_values.tsv", sep="\t", index=False)

    test_name, result = stats(a, b)

    with open(f"{OUTDIR}/{metric}_stats.txt", "w") as f:
        f.write(f"{metric} comparison\n")
        f.write(f"Test: {test_name}\nP-value: {result.pvalue}\n")

    sns.boxplot(data=df, x="Gene", y=metric)
    plt.title(f"{metric}: HepI vs OpsX")
    plt.savefig(f"{OUTDIR}/{metric}.png", dpi=300)
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hepi", required=True)
    parser.add_argument("--opsx", required=True)
    args = parser.parse_args()

    for m in ["GC3", "ENC", "CAI", "RSCU"]:
        run(m, args.hepi, args.opsx)

    logging.info("All analyses completed.")
